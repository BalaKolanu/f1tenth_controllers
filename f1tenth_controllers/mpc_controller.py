#!/usr/bin/env python3
import csv
import math
import os
import time
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - optional fallback for non-ROS test contexts
    get_package_share_directory = None
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Joy
from scipy.sparse import block_diag, csc_matrix
from std_msgs.msg import Bool, Float64
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException, TransformListener
from tf2_ros.buffer import Buffer

from f1tenth_controllers.mpc_utils import nearest_point


@dataclass
class MPCConfig:
    # MPC dimensions and prediction horizon.
    nx: int = 4
    nu: int = 2
    horizon: int = 9

    # Cost weights:
    # r  = input magnitude penalty      [accel, steer]
    # rd = input rate-change penalty    [d(accel), d(steer)]
    # q  = state tracking penalty       [x, y, v, yaw]
    # qf = terminal state penalty       [x, y, v, yaw]
    # Tuned for more aggressive path convergence:
    # - lower steer/steer-rate penalties => willing to steer harder
    # - higher xy/yaw tracking penalties => prioritize lateral/heading alignment
    # - lower speed tracking penalty => allow transient speed tradeoff in corners
    r: np.ndarray = field(default_factory=lambda: np.diag([0.005, 150.0]))
    rd: np.ndarray = field(default_factory=lambda: np.diag([0.0125, 420.0]))
    q: np.ndarray = field(default_factory=lambda: np.diag([90.0, 90.0, 50, 35.0]))
    qf: np.ndarray = field(default_factory=lambda: np.diag([80.0, 80.0, 50, 35.0]))

    # Discrete model and vehicle limits used as hard MPC constraints.
    dt: float = 0.1
    dl: float = 0.03
    wheelbase: float = 0.324
    min_steer: float = -0.4189
    max_steer: float = 0.4189
    max_dsteer: float = np.deg2rad(180.0)
    max_speed: float = 10.0
    min_speed: float = 0.0
    max_accel: float = 6.0
    max_decel: float = 6.0


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0


class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # ROS parameters that are commonly tuned at runtime/launch:
        # - desired_speed_mps / target_speed_mps: nominal speed target
        # - control_rate_hz: control update frequency
        # - mpc_max_steer_rad, mpc_max_dsteer_radps: MPC steering limits
        # - mpc_max_accel_mps2, mpc_max_decel_mps2: longitudinal MPC limits
        # - max_erpm, speed_to_erpm_gain (+ speed clip params): drivetrain speed cap behavior
        # - waypoint_csv / csv_delimiter: path source used by MPC reference generation
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic', 'drive')
        self.declare_parameter('cmd_topic', 'drive')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_tf_pose_feedback', True)
        self.declare_parameter('waypoint_csv', '')
        self.declare_parameter('csv_delimiter', 'auto')
        self.declare_parameter('reverse_path', False)
        self.declare_parameter('ref_path_topic', '/mpc/ref_path')
        self.declare_parameter('pred_path_topic', '/mpc/pred_path')
        self.declare_parameter('state_error_topic', '/mpc/state_error')
        self.declare_parameter('heading_error_topic', '/mpc/heading_error')
        self.declare_parameter('wheelbase_m', 0.324)
        self.declare_parameter('target_speed_mps', 0.45)
        self.declare_parameter('desired_speed_mps', 0.45)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('mpc_max_steer_rad', 0.4189)
        self.declare_parameter('mpc_max_dsteer_radps', float(np.deg2rad(180.0)))
        self.declare_parameter('mpc_max_accel_mps2', 6.0)
        self.declare_parameter('mpc_max_decel_mps2', 6.0)
        self.declare_parameter('odom_speed_alpha', 0.2)
        self.declare_parameter('mpc_osqp_eps_abs', 1e-3)
        self.declare_parameter('mpc_osqp_eps_rel', 1e-3)
        self.declare_parameter('mpc_osqp_max_iter', 20000)
        # Dynamic steering penalty schedule (ERPM-based) from CSV rows:
        # erpm;R_steer;Rd_steer
        # If current ERPM lies between two rows, we linearly interpolate.
        self.declare_parameter('mpc_dynamic_steer_penalty_enable', True)
        self.declare_parameter('mpc_dynamic_steer_penalty_schedule_csv', '')
        self.declare_parameter('max_erpm', 23250.0)
        self.declare_parameter('speed_to_erpm_gain', 4614.0)
        self.declare_parameter('deadman_button_index', 5)
        self.declare_parameter('deadman_button_index_alt', 4)
        self.declare_parameter('deadman_timeout_sec', 0.5)
        self.declare_parameter('deadman_toggle_mode', False)
        self.declare_parameter('speed_clip_axis_index', 7)
        self.declare_parameter('speed_clip_step_erpm', 500.0)
        self.declare_parameter('speed_clip_min_erpm', 500.0)
        # <= 0 disables software upper cap for runtime ERPM clipping.
        self.declare_parameter('speed_clip_max_erpm', 0.0)
        self.declare_parameter('speed_clip_enabled', True)
        self.declare_parameter('speed_clip_toggle_button_index', -1)
        self.declare_parameter('brake_gain_increase_button_index', 2)
        self.declare_parameter('brake_gain_decrease_button_index', 1)
        self.declare_parameter('brake_gain_step_current_per_mps', 0.5)
        self.declare_parameter('speed_clip_controls_target_speed', True)
        self.declare_parameter('launch_speed_floor_mps', 0.0)
        self.declare_parameter('launch_speed_apply_below_mps', 0.15)
        self.declare_parameter('path_index_search_back', 5)
        self.declare_parameter('path_index_search_ahead', 120)
        self.declare_parameter('ref_heading_gate_deg', 120.0)
        self.declare_parameter('ref_reacquire_dist_m', 0.9)
        self.declare_parameter('vesc_active_braking_topic', '/safety/vesc_active_braking_enabled')
        self.declare_parameter('vesc_active_braking_default_enabled', True)
        self.declare_parameter('mpc_direct_brake_enable', True)
        self.declare_parameter('mpc_direct_brake_topic', 'commands/motor/brake')
        self.declare_parameter('mpc_direct_brake_kp_current_per_mps', 15.0)
        self.declare_parameter('mpc_direct_brake_deadband_mps', 0.05)
        self.declare_parameter('mpc_direct_brake_min_current', 0.0)
        self.declare_parameter('mpc_direct_brake_max_current', 30.0)
        self.declare_parameter('emergency_brake_topic', '/safety/emergency_brake')
        self.declare_parameter('recovery_active_topic', '/safety/recovery_active')
        self.declare_parameter('mpc_brake_log_path', self._default_brake_log_path())
        self.declare_parameter('mpc_lap_log_path', self._default_lap_log_path())
        self.declare_parameter('mpc_lap_min_time_sec', 5.0)

        self.odom_topic = str(self.get_parameter('odom_topic').value)
        drive_topic_param = str(self.get_parameter('drive_topic').value)
        cmd_topic_param = str(self.get_parameter('cmd_topic').value)
        self.drive_topic = cmd_topic_param if cmd_topic_param else drive_topic_param
        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.use_tf_pose_feedback = bool(self.get_parameter('use_tf_pose_feedback').value)
        self.waypoint_csv = str(self.get_parameter('waypoint_csv').value)
        self.csv_delimiter = str(self.get_parameter('csv_delimiter').value)
        self.reverse_path = bool(self.get_parameter('reverse_path').value)
        self.ref_path_topic = str(self.get_parameter('ref_path_topic').value)
        self.pred_path_topic = str(self.get_parameter('pred_path_topic').value)
        self.state_error_topic = str(self.get_parameter('state_error_topic').value)
        self.heading_error_topic = str(self.get_parameter('heading_error_topic').value)
        self.wheelbase_m = max(1e-3, float(self.get_parameter('wheelbase_m').value))
        target_speed_mps = float(self.get_parameter('target_speed_mps').value)
        desired_speed_mps = float(self.get_parameter('desired_speed_mps').value)
        self.target_speed_mps = desired_speed_mps if desired_speed_mps > 0.0 else target_speed_mps
        self.control_rate_hz = max(1.0, float(self.get_parameter('control_rate_hz').value))
        self.mpc_max_steer_rad = abs(float(self.get_parameter('mpc_max_steer_rad').value))
        self.mpc_max_dsteer_radps = abs(float(self.get_parameter('mpc_max_dsteer_radps').value))
        self.mpc_max_accel_mps2 = abs(float(self.get_parameter('mpc_max_accel_mps2').value))
        self.mpc_max_decel_mps2 = abs(float(self.get_parameter('mpc_max_decel_mps2').value))
        self.odom_speed_alpha = max(0.0, min(1.0, float(self.get_parameter('odom_speed_alpha').value)))
        self.mpc_osqp_eps_abs = max(1e-6, float(self.get_parameter('mpc_osqp_eps_abs').value))
        self.mpc_osqp_eps_rel = max(1e-6, float(self.get_parameter('mpc_osqp_eps_rel').value))
        self.mpc_osqp_max_iter = max(100, int(self.get_parameter('mpc_osqp_max_iter').value))
        self.mpc_dynamic_steer_penalty_enable = bool(
            self.get_parameter('mpc_dynamic_steer_penalty_enable').value
        )
        schedule_csv_param = str(self.get_parameter('mpc_dynamic_steer_penalty_schedule_csv').value).strip()
        self.mpc_dynamic_steer_penalty_schedule_csv = (
            schedule_csv_param if schedule_csv_param else self._default_steer_penalty_schedule_csv()
        )
        (
            self._steer_penalty_erpm_knots,
            self._steer_penalty_r_knots,
            self._steer_penalty_rd_knots,
        ) = self._load_steer_penalty_schedule(self.mpc_dynamic_steer_penalty_schedule_csv)
        self.max_erpm = float(self.get_parameter('max_erpm').value)
        self.speed_to_erpm_gain = float(self.get_parameter('speed_to_erpm_gain').value)
        self.deadman_button_index = int(self.get_parameter('deadman_button_index').value)
        self.deadman_button_index_alt = int(self.get_parameter('deadman_button_index_alt').value)
        self.deadman_timeout_sec = max(0.0, float(self.get_parameter('deadman_timeout_sec').value))
        self.deadman_toggle_mode = bool(self.get_parameter('deadman_toggle_mode').value)
        self.speed_clip_axis_index = int(self.get_parameter('speed_clip_axis_index').value)
        self.speed_clip_step_erpm = max(1.0, float(self.get_parameter('speed_clip_step_erpm').value))
        self.speed_clip_min_erpm = max(0.0, float(self.get_parameter('speed_clip_min_erpm').value))
        speed_clip_max_raw = float(self.get_parameter('speed_clip_max_erpm').value)
        self.speed_clip_max_erpm = (
            max(self.speed_clip_min_erpm, speed_clip_max_raw)
            if speed_clip_max_raw > 0.0
            else None
        )
        self.speed_clip_enabled = bool(self.get_parameter('speed_clip_enabled').value)
        self.speed_clip_toggle_button_index = int(
            self.get_parameter('speed_clip_toggle_button_index').value
        )
        self.brake_gain_increase_button_index = int(
            self.get_parameter('brake_gain_increase_button_index').value
        )
        self.brake_gain_decrease_button_index = int(
            self.get_parameter('brake_gain_decrease_button_index').value
        )
        self.brake_gain_step_current_per_mps = max(
            0.1,
            float(self.get_parameter('brake_gain_step_current_per_mps').value),
        )
        self.speed_clip_controls_target_speed = bool(
            self.get_parameter('speed_clip_controls_target_speed').value
        )
        self.launch_speed_floor_mps = max(0.0, float(self.get_parameter('launch_speed_floor_mps').value))
        self.launch_speed_apply_below_mps = max(
            0.0, float(self.get_parameter('launch_speed_apply_below_mps').value)
        )
        self.path_index_search_back = max(0, int(self.get_parameter('path_index_search_back').value))
        self.path_index_search_ahead = max(1, int(self.get_parameter('path_index_search_ahead').value))
        self.ref_heading_gate_rad = np.deg2rad(
            max(5.0, min(180.0, float(self.get_parameter('ref_heading_gate_deg').value)))
        )
        self.ref_reacquire_dist_m = max(0.1, float(self.get_parameter('ref_reacquire_dist_m').value))
        self.vesc_active_braking_topic = str(
            self.get_parameter('vesc_active_braking_topic').value
        )
        self.vesc_active_braking_default_enabled = bool(
            self.get_parameter('vesc_active_braking_default_enabled').value
        )
        self.mpc_direct_brake_enable = bool(
            self.get_parameter('mpc_direct_brake_enable').value
        )
        self.mpc_direct_brake_topic = str(
            self.get_parameter('mpc_direct_brake_topic').value
        )
        self.mpc_direct_brake_kp_current_per_mps = max(
            0.0,
            float(self.get_parameter('mpc_direct_brake_kp_current_per_mps').value),
        )
        self.mpc_direct_brake_deadband_mps = max(
            0.0,
            float(self.get_parameter('mpc_direct_brake_deadband_mps').value),
        )
        self.mpc_direct_brake_min_current = max(
            0.0,
            float(self.get_parameter('mpc_direct_brake_min_current').value),
        )
        self.mpc_direct_brake_max_current = max(
            self.mpc_direct_brake_min_current,
            float(self.get_parameter('mpc_direct_brake_max_current').value),
        )
        self.emergency_brake_topic = str(self.get_parameter('emergency_brake_topic').value)
        self.recovery_active_topic = str(self.get_parameter('recovery_active_topic').value)
        self.mpc_brake_log_path = str(self.get_parameter('mpc_brake_log_path').value)
        self.mpc_lap_log_path = str(self.get_parameter('mpc_lap_log_path').value)
        self.mpc_lap_min_time_sec = max(
            0.0,
            float(self.get_parameter('mpc_lap_min_time_sec').value),
        )
        self.max_erpm = max(self.max_erpm, self.speed_clip_min_erpm)
        if self.speed_clip_max_erpm is not None:
            self.max_erpm = min(self.max_erpm, self.speed_clip_max_erpm)

        self.cfg = MPCConfig()
        # Keep MPC discretization consistent with the actual controller update period.
        self.cfg.dt = 1.0 / self.control_rate_hz
        self.cfg.wheelbase = self.wheelbase_m
        self.cfg.max_steer = max(0.05, self.mpc_max_steer_rad)
        self.cfg.min_steer = -self.cfg.max_steer
        self.cfg.max_dsteer = max(0.05, self.mpc_max_dsteer_radps)
        self.cfg.max_accel = max(0.1, self.mpc_max_accel_mps2)
        self.cfg.max_decel = max(0.1, self.mpc_max_decel_mps2)
        # Keep original steer penalties for easy fallback when dynamic schedule is disabled.
        self._base_steer_r = float(self.cfg.r[1, 1])
        self._base_steer_rd = float(self.cfg.rd[1, 1])
        self._last_steer_penalty_segment = None
        self.oa = None
        self.odelta = None

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.ref_path_pub = self.create_publisher(Path, self.ref_path_topic, 10)
        self.pred_path_pub = self.create_publisher(Path, self.pred_path_topic, 10)
        self.state_error_pub = self.create_publisher(Float64, self.state_error_topic, 10)
        self.heading_error_pub = self.create_publisher(Float64, self.heading_error_topic, 10)
        self.vesc_active_braking_pub = self.create_publisher(
            Bool, self.vesc_active_braking_topic, 10
        )
        self.direct_brake_pub = self.create_publisher(Float64, self.mpc_direct_brake_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)
        self.create_subscription(Bool, self.emergency_brake_topic, self.emergency_brake_cb, 10)
        self.create_subscription(Bool, self.recovery_active_topic, self.recovery_active_cb, 10)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state = State()
        self.have_odom = False
        self._filtered_speed_initialized = False
        self.last_ref_index = None
        self.deadman_pressed = False
        self.autonomy_enabled = False
        self._last_deadman_button_pressed = False
        self._last_speed_clip_axis_dir = 0
        self._last_speed_clip_toggle_pressed = False
        self._last_brake_gain_increase_pressed = False
        self._last_brake_gain_decrease_pressed = False
        self._last_joy_msg_monotonic = None
        self._deadman_timeout_warned = False
        self._last_vesc_active_braking_enabled = None
        self.emergency_brake_active = False
        self.recovery_active = False
        self._last_direct_brake_current = 0.0
        self._last_direct_brake_active = False
        self._last_direct_brake_suppressed = False
        self._brake_log_file = None
        self._brake_log_writer = None
        self._lap_count = 0
        self._lap_start_monotonic = None
        self._lap_peak_speed_mps = 0.0
        self._ignore_next_lap = True

        (
            self.cx,
            self.cy,
            self.cyaw,
            self.sp,
            self.path_region_labels,
            self.path_brake_gains,
        ) = self.load_waypoints(self.waypoint_csv, self.csv_delimiter)
        self._lap_path_length_m = self._compute_closed_path_length(self.cx, self.cy)
        self._init_brake_log()
        self._init_lap_log()
        if len(self.cx) < 3:
            raise RuntimeError('Need at least 3 waypoint samples for MPC.')
        if np.any(self.sp > (self.cfg.max_speed + 1e-6)):
            self.get_logger().warn(
                'Waypoint speed profile exceeds MPC max_speed (max sp=%.3f, max_speed=%.3f). '
                'This can increase solver stress due to persistent speed saturation.'
                % (float(np.max(self.sp)), self.cfg.max_speed)
            )

        self.mpc_prob_init()
        # Initialize steer penalties before first solve.
        self._update_dynamic_steer_penalties(0.0, force_log=True)
        self.create_timer(1.0 / self.control_rate_hz, self.control_cb)

        self.get_logger().info(
            'MPC controller ready. odom=%s tf_pose=%s(%s->%s) drive=%s waypoints=%s n=%d dt=%.3f'
            % (
                self.odom_topic,
                str(self.use_tf_pose_feedback).lower(),
                self.map_frame,
                self.base_frame,
                self.drive_topic,
                self.waypoint_csv,
                len(self.cx),
                self.cfg.dt,
            )
        )
        self.get_logger().info(
            'MPC weights: R=[%.3f, %.3f] Rd=[%.3f, %.3f] Q=[%.3f, %.3f, %.3f, %.3f]'
            % (
                self.cfg.r[0, 0],
                self.cfg.r[1, 1],
                self.cfg.rd[0, 0],
                self.cfg.rd[1, 1],
                self.cfg.q[0, 0],
                self.cfg.q[1, 1],
                self.cfg.q[2, 2],
                self.cfg.q[3, 3],
            )
        )
        self.get_logger().info(
            'MPC OSQP settings: eps_abs=%.1e eps_rel=%.1e max_iter=%d'
            % (self.mpc_osqp_eps_abs, self.mpc_osqp_eps_rel, self.mpc_osqp_max_iter)
        )
        self.get_logger().info(
            'Direct MPC braking: enabled=%s kp=%.2f deadband=%.3f current=[%.1f, %.1f]A log=%s'
            % (
                str(self.mpc_direct_brake_enable).lower(),
                self.mpc_direct_brake_kp_current_per_mps,
                self.mpc_direct_brake_deadband_mps,
                self.mpc_direct_brake_min_current,
                self.mpc_direct_brake_max_current,
                self.mpc_brake_log_path,
            )
        )
        self.get_logger().info('MPC lap times log: %s' % self.mpc_lap_log_path)
        self.get_logger().info(
            'Dynamic steer penalties: enabled=%s csv=%s points=%d'
            % (
                str(self.mpc_dynamic_steer_penalty_enable).lower(),
                self.mpc_dynamic_steer_penalty_schedule_csv,
                int(self._steer_penalty_erpm_knots.size),
            )
        )
        knot_summary = ', '.join(
            [
                '%.0f:(%.1f,%.1f)'
                % (e, r, rd)
                for e, r, rd in zip(
                    self._steer_penalty_erpm_knots,
                    self._steer_penalty_r_knots,
                    self._steer_penalty_rd_knots,
                )
            ]
        )
        self.get_logger().info('Dynamic steer penalty knots erpm:(R,Rd): %s' % knot_summary)

    @staticmethod
    def _default_brake_log_path() -> str:
        source_repo = '/home/nvidia/spring26_ws/src/f1tenth_controllers'
        if os.path.isdir(source_repo):
            return os.path.join(source_repo, 'logs', 'mpc_brake_log.csv')
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'logs',
            'mpc_brake_log.csv',
        )

    @staticmethod
    def _default_lap_log_path() -> str:
        source_repo = '/home/nvidia/spring26_ws/src/racestack'
        if os.path.isdir(source_repo):
            return os.path.join(source_repo, 'lap_times.md')
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'lap_times.md',
        )

    def _init_lap_log(self):
        if not self.mpc_lap_log_path:
            return
        try:
            log_dir = os.path.dirname(self.mpc_lap_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            if (
                not os.path.exists(self.mpc_lap_log_path)
                or os.path.getsize(self.mpc_lap_log_path) == 0
            ):
                with open(self.mpc_lap_log_path, 'w', encoding='utf-8') as log_file:
                    log_file.write('# MPC Lap Times\n\n')
                    log_file.write(
                        '| Lap | Lap time (s) | Avg speed (m/s) | Peak speed (m/s) | Ref index |\n'
                    )
                    log_file.write('| ---: | ---: | ---: | ---: | ---: |\n')
        except OSError as exc:
            self.get_logger().warning(
                'Failed to initialize MPC lap log %s: %s'
                % (self.mpc_lap_log_path, str(exc))
            )

    def _maybe_record_lap(self, previous_index, current_index: int, ncourse: int, speed_mps: float):
        if not self.mpc_lap_log_path:
            return
        now_monotonic = time.monotonic()
        current_speed = abs(float(speed_mps))
        if self._lap_start_monotonic is None:
            self._lap_start_monotonic = now_monotonic
            self._lap_peak_speed_mps = current_speed
            return
        self._lap_peak_speed_mps = max(self._lap_peak_speed_mps, current_speed)
        if previous_index is None or ncourse <= 4:
            return

        previous_index = int(previous_index)
        current_index = int(current_index)
        crossed_start = previous_index > int(0.75 * ncourse) and current_index < int(0.25 * ncourse)
        if not crossed_start:
            return

        lap_time = now_monotonic - self._lap_start_monotonic
        if lap_time < self.mpc_lap_min_time_sec:
            return

        if self._ignore_next_lap:
            self._ignore_next_lap = False
            self._lap_start_monotonic = now_monotonic
            self._lap_peak_speed_mps = current_speed
            self.get_logger().info('Ignoring first completed lap; lap timing starts now.')
            return

        self._lap_count += 1
        avg_speed_mps = self._lap_path_length_m / lap_time if lap_time > 1e-6 else 0.0
        peak_speed_mps = self._lap_peak_speed_mps
        self._lap_start_monotonic = now_monotonic
        self._lap_peak_speed_mps = current_speed
        try:
            with open(self.mpc_lap_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(
                    '| %d | %.3f | %.3f | %.3f | %d |\n'
                    % (self._lap_count, lap_time, avg_speed_mps, peak_speed_mps, current_index)
                )
            self.get_logger().info(
                'Lap %d recorded: %.3f s avg=%.3f m/s peak=%.3f m/s'
                % (self._lap_count, lap_time, avg_speed_mps, peak_speed_mps)
            )
        except OSError as exc:
            self.get_logger().warning('Failed to write MPC lap log: %s' % str(exc))

    def _init_brake_log(self):
        if not self.mpc_brake_log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.mpc_brake_log_path), exist_ok=True)
            file_exists = os.path.exists(self.mpc_brake_log_path)
            self._brake_log_file = open(self.mpc_brake_log_path, 'a', newline='', buffering=1)
            self._brake_log_writer = csv.writer(self._brake_log_file)
            if (not file_exists) or os.path.getsize(self.mpc_brake_log_path) == 0:
                self._brake_log_writer.writerow([
                    'wall_time_sec',
                    'ros_time_ns',
                    'target_speed_mps',
                    'measured_speed_mps',
                    'speed_error_mps',
                    'accel_cmd_mps2',
                    'brake_current_a',
                    'ref_speed_mps',
                    'ref_index',
                    'emergency_brake_active',
                    'recovery_active',
                ])
        except OSError as exc:
            self.get_logger().warning(
                'Failed to open MPC brake log %s: %s' % (self.mpc_brake_log_path, str(exc))
            )
            self._brake_log_file = None
            self._brake_log_writer = None

    def _log_brake_sample(
        self,
        target_speed_mps: float,
        measured_speed_mps: float,
        accel_cmd_mps2: float,
        brake_current_a: float,
        ref_speed_mps: float,
    ):
        if self._brake_log_writer is None:
            return
        speed_error_mps = max(0.0, measured_speed_mps - target_speed_mps)
        try:
            self._brake_log_writer.writerow([
                '%.6f' % time.time(),
                int(self.get_clock().now().nanoseconds),
                '%.6f' % target_speed_mps,
                '%.6f' % measured_speed_mps,
                '%.6f' % speed_error_mps,
                '%.6f' % accel_cmd_mps2,
                '%.6f' % brake_current_a,
                '%.6f' % ref_speed_mps,
                int(self.last_ref_index) if self.last_ref_index is not None else -1,
                int(self.emergency_brake_active),
                int(self.recovery_active),
            ])
        except OSError as exc:
            self.get_logger().warning('Failed to write MPC brake log: %s' % str(exc))
            self._brake_log_writer = None
            if self._brake_log_file is not None:
                self._brake_log_file.close()
                self._brake_log_file = None

    def _default_steer_penalty_schedule_csv(self) -> str:
        filename = 'mpc_steer_penalty_schedule.csv'
        if get_package_share_directory is not None:
            try:
                share_dir = get_package_share_directory('f1tenth_controllers')
                return os.path.join(share_dir, 'config', filename)
            except Exception:
                pass
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', filename)

    def _load_steer_penalty_schedule(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise RuntimeError('Steer penalty schedule CSV not found: %s' % csv_path)

        table = np.genfromtxt(csv_path, delimiter=';', comments='#', invalid_raise=False)
        if table.ndim == 1:
            table = table.reshape(1, -1)
        if table.shape[1] < 3:
            table = np.genfromtxt(csv_path, delimiter=',', comments='#', invalid_raise=False)
            if table.ndim == 1:
                table = table.reshape(1, -1)
        if table.shape[1] < 3:
            raise RuntimeError(
                'Steer penalty schedule CSV must have at least 3 columns: erpm;r;rd. File: %s'
                % csv_path
            )

        table = np.asarray(table[:, :3], dtype=np.float64)
        table = table[np.all(np.isfinite(table), axis=1)]
        if table.shape[0] < 2:
            raise RuntimeError(
                'Steer penalty schedule CSV must have at least 2 numeric rows. File: %s'
                % csv_path
            )

        # Sort by ERPM and collapse duplicates (last value wins).
        order = np.argsort(table[:, 0], kind='stable')
        table = table[order]
        schedule = {}
        for row in table:
            erpm = max(0.0, float(row[0]))
            r = max(0.0, float(row[1]))
            rd = max(0.0, float(row[2]))
            schedule[erpm] = (r, rd)

        erpm_knots = np.array(sorted(schedule.keys()), dtype=np.float64)
        r_knots = np.array([schedule[e][0] for e in erpm_knots], dtype=np.float64)
        rd_knots = np.array([schedule[e][1] for e in erpm_knots], dtype=np.float64)
        if erpm_knots.size < 2:
            raise RuntimeError(
                'Steer penalty schedule CSV must resolve to at least 2 distinct ERPM knots. File: %s'
                % csv_path
            )
        return erpm_knots, r_knots, rd_knots

    def _set_input_cost_params(self):
        r_stage = np.array(
            [max(0.0, float(self.cfg.r[0, 0])), max(0.0, float(self.cfg.r[1, 1]))],
            dtype=np.float64,
        )
        self.r_diag_k.value = np.tile(r_stage, self.cfg.horizon)
        if self.cfg.horizon > 1:
            rd_stage = np.array(
                [max(0.0, float(self.cfg.rd[0, 0])), max(0.0, float(self.cfg.rd[1, 1]))],
                dtype=np.float64,
            )
            self.rd_diag_k.value = np.tile(rd_stage, self.cfg.horizon - 1)

    def _update_dynamic_steer_penalties(self, speed_mps: float, force_log: bool = False):
        if not self.mpc_dynamic_steer_penalty_enable:
            self.cfg.r[1, 1] = self._base_steer_r
            self.cfg.rd[1, 1] = self._base_steer_rd
            self._set_input_cost_params()
            return

        erpm_abs = 0.0
        if abs(self.speed_to_erpm_gain) > 1e-6:
            erpm_abs = abs(float(speed_mps)) * abs(self.speed_to_erpm_gain)

        knots = self._steer_penalty_erpm_knots
        r_knots = self._steer_penalty_r_knots
        rd_knots = self._steer_penalty_rd_knots

        if erpm_abs <= knots[0]:
            r_steer = float(r_knots[0])
            rd_steer = float(rd_knots[0])
            segment = ('<=', float(knots[0]))
        elif erpm_abs >= knots[-1]:
            r_steer = float(r_knots[-1])
            rd_steer = float(rd_knots[-1])
            segment = ('>=', float(knots[-1]))
        else:
            hi = int(np.searchsorted(knots, erpm_abs, side='right'))
            lo = hi - 1
            e0 = float(knots[lo])
            e1 = float(knots[hi])
            t = (erpm_abs - e0) / max(1e-9, (e1 - e0))
            r_steer = float((1.0 - t) * r_knots[lo] + t * r_knots[hi])
            rd_steer = float((1.0 - t) * rd_knots[lo] + t * rd_knots[hi])
            segment = (e0, e1)

        self.cfg.r[1, 1] = float(max(0.0, r_steer))
        self.cfg.rd[1, 1] = float(max(0.0, rd_steer))
        self._set_input_cost_params()

        if force_log or (self._last_steer_penalty_segment != segment):
            self._last_steer_penalty_segment = segment
            self.get_logger().debug(
                'Dynamic steer penalty update: erpm_est=%.0f segment=%s Rsteer=%.1f Rdsteer=%.1f'
                % (erpm_abs, str(segment), self.cfg.r[1, 1], self.cfg.rd[1, 1])
            )

    def load_waypoints(self, csv_path: str, delimiter: str):
        if not os.path.exists(csv_path):
            self.get_logger().error('Waypoint CSV not found: %s' % csv_path)
            return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])

        delim = delimiter
        if delim == 'auto':
            delim = ','
            with open(csv_path, 'r') as csv_file:
                first_line = csv_file.readline()
                if ';' in first_line:
                    delim = ';'

        # Robustly handle CSVs with extra non-numeric columns by selecting only needed fields.
        data = np.genfromtxt(csv_path, delimiter=delim, comments='#', invalid_raise=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])

        if delim == ';' and data.shape[1] >= 6:
            cx_list = []
            cy_list = []
            cyaw_list = []
            sp_list = []
            labels = []
            brake_gain_list = []
            with open(csv_path, 'r', newline='') as csv_file:
                first_row = next(csv.reader(csv_file, delimiter=';'), [])
            header_names = [cell.strip() for cell in first_row]
            has_named_header = {'x', 'y', 'yaw'}.issubset(set(header_names))
            if has_named_header:
                with open(csv_path, 'r', newline='') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=';')
                    for row in reader:
                        if not row:
                            continue
                        first_value = next(iter(row.values()), '')
                        if str(first_value).strip().startswith('#'):
                            continue
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                            yaw = float(row['yaw'])
                            raw_speed = row.get('v', '')
                            speed = (
                                abs(float(raw_speed))
                                if raw_speed is not None and str(raw_speed).strip() != ''
                                else self.target_speed_mps
                            )
                        except (TypeError, ValueError, KeyError):
                            continue
                        if not (
                            np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw) and np.isfinite(speed)
                        ):
                            continue
                        brake_gain = np.nan
                        raw_brake_gain = row.get('brake_gain', '')
                        if raw_brake_gain is not None and str(raw_brake_gain).strip() != '':
                            try:
                                brake_gain = max(0.0, float(raw_brake_gain))
                            except (TypeError, ValueError):
                                brake_gain = np.nan
                        cx_list.append(x)
                        cy_list.append(y)
                        cyaw_list.append(yaw)
                        sp_list.append(speed)
                        labels.append(
                            str(row.get('turn_state', row.get('segment', ''))).strip().upper()
                        )
                        brake_gain_list.append(brake_gain)
            else:
                with open(csv_path, 'r', newline='') as csv_file:
                    reader = csv.reader(csv_file, delimiter=';')
                    for row in reader:
                        if not row or row[0].strip().startswith('#') or len(row) < 6:
                            continue
                        try:
                            x = float(row[1])
                            y = float(row[2])
                            yaw = float(row[3])
                            speed = abs(float(row[5]))
                        except (TypeError, ValueError):
                            continue
                        if not (
                            np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw) and np.isfinite(speed)
                        ):
                            continue
                        cx_list.append(x)
                        cy_list.append(y)
                        cyaw_list.append(yaw)
                        sp_list.append(speed)
                        labels.append(row[6].strip().upper() if len(row) > 6 else '')
                        brake_gain_list.append(np.nan)
            cx = np.asarray(cx_list, dtype=np.float64)
            cy = np.asarray(cy_list, dtype=np.float64)
            cyaw = np.asarray(cyaw_list, dtype=np.float64)
            sp = np.asarray(sp_list, dtype=np.float64)
            brake_gains = np.asarray(brake_gain_list, dtype=np.float64)
        else:
            # Generic XY CSV (optionally with extra columns). Use first two columns only.
            xy = np.genfromtxt(
                csv_path,
                delimiter=delim,
                comments='#',
                usecols=(0, 1),
                invalid_raise=False,
            )
            if xy.ndim == 1:
                xy = xy.reshape(1, 2)
            if xy.shape[0] == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])
            finite_rows = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[finite_rows]
            if xy.shape[0] < 3:
                return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])
            cx = xy[:, 0]
            cy = xy[:, 1]
            seg_dx = np.diff(np.r_[cx, cx[0]])
            seg_dy = np.diff(np.r_[cy, cy[0]])
            cyaw = np.arctan2(seg_dy, seg_dx)
            sp = np.full_like(cx, 10.0)
            labels = [''] * int(cx.shape[0])
            brake_gains = np.full(int(cx.shape[0]), np.nan, dtype=np.float64)

        if cx.shape[0] < 3:
            return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])

        if self.reverse_path:
            cx = cx[::-1].copy()
            cy = cy[::-1].copy()
            sp = sp[::-1].copy()
            labels = list(reversed(labels))
            brake_gains = brake_gains[::-1].copy()
            cyaw = self._compute_path_yaw(cx, cy)

        if len(labels) != int(cx.shape[0]):
            labels = [''] * int(cx.shape[0])
        if brake_gains.shape[0] != int(cx.shape[0]):
            brake_gains = np.full(int(cx.shape[0]), np.nan, dtype=np.float64)

        return cx, cy, cyaw, sp, labels, brake_gains

    def current_brake_gain_kp(self) -> float:
        if self.last_ref_index is not None and self.path_brake_gains.shape[0] > int(self.last_ref_index):
            csv_gain = float(self.path_brake_gains[int(self.last_ref_index)])
            if np.isfinite(csv_gain):
                return max(0.0, csv_gain)
        return self.mpc_direct_brake_kp_current_per_mps

    @staticmethod
    def _compute_closed_path_length(cx: np.ndarray, cy: np.ndarray) -> float:
        if cx.shape[0] < 2 or cy.shape[0] < 2:
            return 0.0
        dx = np.diff(np.r_[cx, cx[0]])
        dy = np.diff(np.r_[cy, cy[0]])
        return float(np.sum(np.hypot(dx, dy)))

    @staticmethod
    def _compute_path_yaw(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        if cx.shape[0] < 2:
            return np.array([], dtype=np.float64)
        seg_dx = np.diff(np.r_[cx, cx[0]])
        seg_dy = np.diff(np.r_[cy, cy[0]])
        return np.arctan2(seg_dy, seg_dx)

    def odom_cb(self, msg: Odometry):
        # Odom is used as velocity feedback; pose can optionally come from TF (map->base_link).
        measured_speed = float(msg.twist.twist.linear.x)
        if self._filtered_speed_initialized:
            self.state.v = (
                self.odom_speed_alpha * measured_speed
                + (1.0 - self.odom_speed_alpha) * self.state.v
            )
        else:
            self.state.v = measured_speed
            self._filtered_speed_initialized = True
        if not self.use_tf_pose_feedback:
            self.state.x = msg.pose.pose.position.x
            self.state.y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.state.yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
        self.have_odom = True

    def emergency_brake_cb(self, msg: Bool):
        self.emergency_brake_active = bool(msg.data)

    def recovery_active_cb(self, msg: Bool):
        self.recovery_active = bool(msg.data)

    def compute_direct_brake_current(self, target_speed: float, accel_cmd: float) -> float:
        if not self.mpc_direct_brake_enable:
            return 0.0
        if self.emergency_brake_active or self.recovery_active:
            return 0.0
        if accel_cmd >= 0.0:
            return 0.0
        speed_error = float(self.state.v) - float(target_speed)
        if speed_error <= self.mpc_direct_brake_deadband_mps:
            return 0.0
        raw = self.current_brake_gain_kp() * (
            speed_error - self.mpc_direct_brake_deadband_mps
        )
        if raw <= 0.0:
            return 0.0
        brake_current = min(self.mpc_direct_brake_max_current, raw)
        if brake_current > 0.0:
            brake_current = max(self.mpc_direct_brake_min_current, brake_current)
        return brake_current

    def publish_direct_brake_current(self, brake_current: float):
        if not self.mpc_direct_brake_enable:
            return
        command = max(0.0, float(brake_current))
        if command <= 0.0 and self._last_direct_brake_current <= 0.0:
            return
        msg = Float64()
        msg.data = command
        self.direct_brake_pub.publish(msg)
        self._last_direct_brake_current = command

    def control_cb(self):
        if not self.have_odom:
            return
        if self.use_tf_pose_feedback:
            pose = self.lookup_pose_tf()
            if pose is None:
                self.publish_stop()
                return
            self.state.x, self.state.y, self.state.yaw = pose

        ref = self.calc_ref_trajectory(self.state, self.cx, self.cy, self.cyaw, self.sp)
        self.publish_vesc_active_braking_state()
        self.publish_tracking_errors(ref)
        if not self.deadman_active():
            self.publish_stop()
            return

        # Keep initial speed state consistent with MPC hard bounds to avoid infeasible QP.
        v0 = min(self.cfg.max_speed, max(self.cfg.min_speed, self.state.v))
        x0 = [self.state.x, self.state.y, v0, self.state.yaw]
        self.oa, self.odelta, mpc_x, mpc_y, mpc_yaw, _, _ = self.linear_mpc_control(ref, x0, self.oa, self.odelta)
        self.publish_path(self.ref_path_pub, ref[0, :], ref[1, :], ref[3, :], self.map_frame)
        self.publish_path(self.pred_path_pub, mpc_x, mpc_y, mpc_yaw, self.map_frame)

        steer = float(self.odelta[0]) if self.odelta is not None else 0.0
        accel = float(self.oa[0]) if self.oa is not None else 0.0
        speed = float(self.state.v + accel * self.cfg.dt)
        if (
            self.speed_clip_enabled
            and self.speed_clip_controls_target_speed
            and abs(self.speed_to_erpm_gain) > 1e-6
        ):
            max_speed_from_erpm = self.max_erpm / abs(self.speed_to_erpm_gain)
            speed = min(speed, max_speed_from_erpm)
        # Optional launch assist to overcome drivetrain static friction from standstill.
        if (
            self.launch_speed_floor_mps > 0.0
            and speed > 0.0
            and abs(self.state.v) < self.launch_speed_apply_below_mps
        ):
            speed = max(speed, self.launch_speed_floor_mps)
        speed = max(0.0, speed)
        brake_current = self.compute_direct_brake_current(speed, accel)

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = steer
        cmd.drive.speed = speed
        # Direct brake current is handled in this controller, so the Ackermann
        # acceleration field is used only as a propulsion hint.
        cmd.drive.acceleration = max(0.0, accel)
        self.drive_pub.publish(cmd)
        if self.emergency_brake_active:
            if self._last_direct_brake_current > 0.0 and not self._last_direct_brake_suppressed:
                self.get_logger().info('Direct MPC braking suppressed by emergency brake')
                self._last_direct_brake_suppressed = True
            self._last_direct_brake_active = False
            self._last_direct_brake_current = 0.0
            return
        if self.recovery_active:
            if self._last_direct_brake_current > 0.0 and not self._last_direct_brake_suppressed:
                self.get_logger().info('Direct MPC braking suppressed by collision recovery')
                self._last_direct_brake_suppressed = True
            self.publish_direct_brake_current(0.0)
            self._last_direct_brake_active = False
            return
        self._last_direct_brake_suppressed = False
        self.publish_direct_brake_current(brake_current)
        brake_active = brake_current > 0.0
        if brake_active and not self._last_direct_brake_active:
            self.get_logger().info(
                'Direct MPC braking engaged: target=%.2f measured=%.2f current=%.2f kp=%.2f'
                % (speed, self.state.v, brake_current, self.current_brake_gain_kp())
            )
        elif (not brake_active) and self._last_direct_brake_active:
            self.get_logger().info(
                'Direct MPC braking released: target=%.2f measured=%.2f'
                % (speed, self.state.v)
            )
        self._last_direct_brake_active = brake_active
        if brake_active:
            ref_speed = float(ref[2, 0]) if ref is not None and ref.shape[1] > 0 else 0.0
            self._log_brake_sample(speed, float(self.state.v), accel, brake_current, ref_speed)

    def publish_vesc_active_braking_state(self):
        enabled = False
        if self._last_vesc_active_braking_enabled == enabled:
            return
        self._last_vesc_active_braking_enabled = enabled
        msg = Bool()
        msg.data = enabled
        self.vesc_active_braking_pub.publish(msg)
        self.get_logger().info(
            'VESC active braking %s (disabled; MPC direct brake controller owns braking)'
            % (
                'ENABLED' if enabled else 'disabled',
            )
        )

    @staticmethod
    def region_label_is_turn(label: str) -> bool:
        return label.strip().upper() != 'S'

    def lookup_pose_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        x = float(t.transform.translation.x)
        y = float(t.transform.translation.y)
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (x, y, yaw)

    def joy_cb(self, msg: Joy):
        self._last_joy_msg_monotonic = time.monotonic()
        pressed_primary = (
            0 <= self.deadman_button_index < len(msg.buttons)
            and msg.buttons[self.deadman_button_index] == 1
        )
        pressed_alt = (
            0 <= self.deadman_button_index_alt < len(msg.buttons)
            and msg.buttons[self.deadman_button_index_alt] == 1
        )
        pressed_now = bool(pressed_primary or pressed_alt)
        self.deadman_pressed = pressed_now

        if self.deadman_toggle_mode:
            if pressed_now and not self._last_deadman_button_pressed:
                self.autonomy_enabled = not self.autonomy_enabled
                state = 'ENABLED' if self.autonomy_enabled else 'DISABLED'
                self.get_logger().info('Autonomy toggle switched: %s' % state)
            self._last_deadman_button_pressed = pressed_now

        speed_clip_toggle_pressed = (
            0 <= self.speed_clip_toggle_button_index < len(msg.buttons)
            and msg.buttons[self.speed_clip_toggle_button_index] == 1
        )
        if speed_clip_toggle_pressed and not self._last_speed_clip_toggle_pressed:
            self.speed_clip_enabled = not self.speed_clip_enabled
            state = 'ENABLED' if self.speed_clip_enabled else 'DISABLED'
            self.get_logger().info('MPC speed clip toggled: %s' % state)
        self._last_speed_clip_toggle_pressed = speed_clip_toggle_pressed

        axis_dir = 0
        if 0 <= self.speed_clip_axis_index < len(msg.axes):
            axis_val = float(msg.axes[self.speed_clip_axis_index])
            if axis_val > 0.5:
                axis_dir = 1
            elif axis_val < -0.5:
                axis_dir = -1

        if axis_dir != 0 and axis_dir != self._last_speed_clip_axis_dir:
            delta = self.speed_clip_step_erpm if axis_dir > 0 else -self.speed_clip_step_erpm
            self.max_erpm = max(self.speed_clip_min_erpm, self.max_erpm + delta)
            if self.speed_clip_max_erpm is not None:
                self.max_erpm = min(self.speed_clip_max_erpm, self.max_erpm)
            self.get_logger().info('Updated runtime speed clip from axis %d: max_erpm=%.0f' % (
                self.speed_clip_axis_index, self.max_erpm
            ))
        self._last_speed_clip_axis_dir = axis_dir

        inc_pressed = (
            0 <= self.brake_gain_increase_button_index < len(msg.buttons)
            and msg.buttons[self.brake_gain_increase_button_index] == 1
        )
        dec_pressed = (
            0 <= self.brake_gain_decrease_button_index < len(msg.buttons)
            and msg.buttons[self.brake_gain_decrease_button_index] == 1
        )

        if inc_pressed and not self._last_brake_gain_increase_pressed:
            self.mpc_direct_brake_kp_current_per_mps += self.brake_gain_step_current_per_mps
            self.get_logger().info(
                'Increased MPC brake gain from button %d: kp=%.2f'
                % (
                    self.brake_gain_increase_button_index,
                    self.mpc_direct_brake_kp_current_per_mps,
                )
            )
        if dec_pressed and not self._last_brake_gain_decrease_pressed:
            self.mpc_direct_brake_kp_current_per_mps = max(
                0.0,
                self.mpc_direct_brake_kp_current_per_mps - self.brake_gain_step_current_per_mps,
            )
            self.get_logger().info(
                'Decreased MPC brake gain from button %d: kp=%.2f'
                % (
                    self.brake_gain_decrease_button_index,
                    self.mpc_direct_brake_kp_current_per_mps,
                )
            )
        self._last_brake_gain_increase_pressed = inc_pressed
        self._last_brake_gain_decrease_pressed = dec_pressed

    def deadman_active(self) -> bool:
        if self.deadman_toggle_mode:
            return self.autonomy_enabled
        if self.deadman_timeout_sec > 0.0:
            if self._last_joy_msg_monotonic is None:
                return False
            joy_age = time.monotonic() - self._last_joy_msg_monotonic
            if joy_age > self.deadman_timeout_sec:
                if not self._deadman_timeout_warned:
                    self.get_logger().warn(
                        'Deadman timeout: no /joy update for %.3fs (limit %.3fs)'
                        % (joy_age, self.deadman_timeout_sec)
                    )
                    self._deadman_timeout_warned = True
                return False
            self._deadman_timeout_warned = False
        return self.deadman_pressed

    def publish_stop(self):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = 0.0
        cmd.drive.speed = 0.0
        cmd.drive.acceleration = 0.0
        self.drive_pub.publish(cmd)

    def destroy_node(self):
        if self._brake_log_file is not None:
            self._brake_log_file.close()
            self._brake_log_file = None
            self._brake_log_writer = None
        return super().destroy_node()

    def publish_path(self, pub, xs, ys, yaws, frame_id: str):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        for x, y, yaw in zip(xs, ys, yaws):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            half = 0.5 * float(yaw)
            pose.pose.orientation.w = math.cos(half)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(half)
            msg.poses.append(pose)
        pub.publish(msg)

    def publish_tracking_errors(self, ref):
        if ref is None or ref.shape[1] == 0:
            return

        dx = float(ref[0, 0]) - float(self.state.x)
        dy = float(ref[1, 0]) - float(self.state.y)
        state_error = math.hypot(dx, dy)
        heading_error = float(self.wrap_to_pi(float(ref[3, 0]) - float(self.state.yaw)))

        state_msg = Float64()
        state_msg.data = state_error
        self.state_error_pub.publish(state_msg)

        heading_msg = Float64()
        heading_msg.data = heading_error
        self.heading_error_pub.publish(heading_msg)

    def mpc_prob_init(self):
        self.xk = cvxpy.Variable((self.cfg.nx, self.cfg.horizon + 1))
        self.uk = cvxpy.Variable((self.cfg.nu, self.cfg.horizon))

        self.x0k = cvxpy.Parameter((self.cfg.nx,))
        self.x0k.value = np.zeros((self.cfg.nx,))

        self.ref_traj_k = cvxpy.Parameter((self.cfg.nx, self.cfg.horizon + 1))
        self.ref_traj_k.value = np.zeros((self.cfg.nx, self.cfg.horizon + 1))

        q_block = [self.cfg.q] * self.cfg.horizon
        q_block.append(self.cfg.qf)
        q_block = block_diag(tuple(q_block))

        self.r_diag_k = cvxpy.Parameter((self.cfg.nu * self.cfg.horizon,), nonneg=True)
        if self.cfg.horizon > 1:
            self.rd_diag_k = cvxpy.Parameter((self.cfg.nu * (self.cfg.horizon - 1),), nonneg=True)
        self._set_input_cost_params()

        objective = 0.0
        constraints = []

        objective += cvxpy.sum(cvxpy.multiply(self.r_diag_k, cvxpy.square(cvxpy.vec(self.uk))))
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), q_block)
        if self.cfg.horizon > 1:
            objective += cvxpy.sum(
                cvxpy.multiply(
                    self.rd_diag_k,
                    cvxpy.square(cvxpy.vec(cvxpy.diff(self.uk, axis=1))),
                )
            )

        A_block = []
        B_block = []
        C_block = []
        path_predict = np.zeros((self.cfg.nx, self.cfg.horizon + 1))
        for t in range(self.cfg.horizon):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))
        self.Annz_k.value = A_block.data
        self.Ak_ = cvxpy.reshape(indexer @ self.Annz_k, (m, n), order='C')

        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bnnz_k.value = B_block.data
        self.Bk_ = cvxpy.reshape(indexer @ self.Bnnz_k, (m, n), order='C')

        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        flatten_prev_x = cvxpy.vec(self.xk[:, :-1])
        flatten_next_x = cvxpy.vec(self.xk[:, 1:])
        constraints.append(flatten_next_x == self.Ak_ @ flatten_prev_x + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_)

        dsteering = cvxpy.diff(self.uk[1, :])
        constraints.append(-self.cfg.max_dsteer * self.cfg.dt <= dsteering)
        constraints.append(dsteering <= self.cfg.max_dsteer * self.cfg.dt)

        constraints.append(self.xk[:, 0] == self.x0k)
        constraints.append(self.cfg.min_speed <= self.xk[2, :])
        constraints.append(self.xk[2, :] <= self.cfg.max_speed)
        constraints.append(self.cfg.min_steer <= self.uk[1, :])
        constraints.append(self.uk[1, :] <= self.cfg.max_steer)
        constraints.append(-self.cfg.max_decel <= self.uk[0, :])
        constraints.append(self.uk[0, :] <= self.cfg.max_accel)

        self.mpc_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        ref = np.zeros((self.cfg.nx, self.cfg.horizon + 1))
        ncourse = len(cx)
        cyaw = np.copy(cyaw)
        ind = self.find_reference_index(state, cx, cy, cyaw)
        self._maybe_record_lap(self.last_ref_index, int(ind), ncourse, state.v)
        self.last_ref_index = int(ind)

        travel = abs(state.v) * self.cfg.dt
        dind = max(1, int(max(1e-6, travel / self.cfg.dl)))
        ind_list = int(ind) + np.insert(np.cumsum(np.repeat(dind, self.cfg.horizon)), 0, 0).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse

        ref[0, :] = cx[ind_list]
        ref[1, :] = cy[ind_list]
        ref[2, :] = sp[ind_list]
        ref_yaw = cyaw[ind_list]
        ref[3, :] = state.yaw + self.wrap_to_pi(ref_yaw - state.yaw)
        return ref

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def find_reference_index(self, state, cx, cy, cyaw):
        def _global_best_index():
            dxg = cx - float(state.x)
            dyg = cy - float(state.y)
            d2g = dxg * dxg + dyg * dyg
            heading_err_g = np.abs(self.wrap_to_pi(cyaw - float(state.yaw)))
            aligned_g = heading_err_g <= self.ref_heading_gate_rad
            if np.any(aligned_g):
                c = np.where(aligned_g)[0]
                return int(c[int(np.argmin(d2g[c]))])
            return int(np.argmin(d2g))

        ncourse = len(cx)
        if self.last_ref_index is None or ncourse <= 2:
            return _global_best_index()

        offsets = np.arange(-self.path_index_search_back, self.path_index_search_ahead + 1, dtype=np.int64)
        cand = (int(self.last_ref_index) + offsets) % ncourse

        dx = cx[cand] - float(state.x)
        dy = cy[cand] - float(state.y)
        d2 = dx * dx + dy * dy

        heading_err = np.abs(self.wrap_to_pi(cyaw[cand] - float(state.yaw)))
        aligned = heading_err <= self.ref_heading_gate_rad
        if np.any(aligned):
            cand = cand[aligned]
            d2 = d2[aligned]

        best_local = int(np.argmin(d2))
        best_ind = int(cand[best_local])

        # If local-window association drifts too far, reacquire globally.
        if float(np.sqrt(d2[best_local])) > self.ref_reacquire_dist_m:
            return _global_best_index()
        return best_ind

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, self.cfg.horizon + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    def update_state(self, state, a, delta):
        delta = max(self.cfg.min_steer, min(self.cfg.max_steer, delta))

        state.x = state.x + state.v * math.cos(state.yaw) * self.cfg.dt
        state.y = state.y + state.v * math.sin(state.yaw) * self.cfg.dt
        state.yaw = state.yaw + (state.v / self.cfg.wheelbase) * math.tan(delta) * self.cfg.dt
        state.v = state.v + a * self.cfg.dt

        if state.v > self.cfg.max_speed:
            state.v = self.cfg.max_speed
        elif state.v < self.cfg.min_speed:
            state.v = self.cfg.min_speed

        return state

    def get_model_matrix(self, v, phi, delta):
        A = np.zeros((self.cfg.nx, self.cfg.nx))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.cfg.dt * math.cos(phi)
        A[0, 3] = -self.cfg.dt * v * math.sin(phi)
        A[1, 2] = self.cfg.dt * math.sin(phi)
        A[1, 3] = self.cfg.dt * v * math.cos(phi)
        A[3, 2] = self.cfg.dt * math.tan(delta) / self.cfg.wheelbase

        B = np.zeros((self.cfg.nx, self.cfg.nu))
        B[2, 0] = self.cfg.dt
        B[3, 1] = self.cfg.dt * v / (self.cfg.wheelbase * math.cos(delta) ** 2)

        C = np.zeros(self.cfg.nx)
        C[0] = self.cfg.dt * v * math.sin(phi) * phi
        C[1] = -self.cfg.dt * v * math.cos(phi) * phi
        C[3] = -self.cfg.dt * v * delta / (self.cfg.wheelbase * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        x0_arr = np.asarray(x0, dtype=np.float64)
        ref_arr = np.asarray(ref_traj, dtype=np.float64)
        pred_arr = np.asarray(path_predict, dtype=np.float64)
        if (
            not np.all(np.isfinite(x0_arr))
            or not np.all(np.isfinite(ref_arr))
            or not np.all(np.isfinite(pred_arr))
        ):
            self.get_logger().warn('Skipping MPC solve due to non-finite inputs.')
            return None, None, None, None, None, None

        self.x0k.value = x0_arr
        self._update_dynamic_steer_penalties(float(x0_arr[2]))

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.cfg.horizon):
            A, B, C = self.get_model_matrix(pred_arr[2, t], pred_arr[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block
        self.ref_traj_k.value = ref_arr

        try:
            self.mpc_prob.solve(
                solver=cvxpy.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=self.mpc_osqp_eps_abs,
                eps_rel=self.mpc_osqp_eps_rel,
                max_iter=self.mpc_osqp_max_iter,
            )
        except cvxpy.error.SolverError as exc:
            self.get_logger().warn('OSQP warm-start solve failed (%s); retrying cold-start.' % str(exc))
            try:
                self.mpc_prob.solve(
                    solver=cvxpy.OSQP,
                    warm_start=False,
                    verbose=False,
                    eps_abs=self.mpc_osqp_eps_abs,
                    eps_rel=self.mpc_osqp_eps_rel,
                    max_iter=self.mpc_osqp_max_iter,
                )
            except cvxpy.error.SolverError as exc_retry:
                self.get_logger().warn('OSQP retry failed (%s); using safe fallback this cycle.' % str(exc_retry))
                return None, None, None, None, None, None

        if self.mpc_prob.status in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
            return oa, odelta, ox, oy, oyaw, ov

        self.get_logger().warn('MPC solve failed: %s' % self.mpc_prob.status)
        return None, None, None, None, None, None

    def linear_mpc_control(self, ref_path, x0, oa, od):
        if oa is None or od is None:
            oa = [0.0] * self.cfg.horizon
            od = [0.0] * self.cfg.horizon

        path_predict = self.predict_motion(x0, oa, od, ref_path)
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(ref_path, path_predict, x0)

        if mpc_a is None:
            mpc_a = [0.0] * self.cfg.horizon
            mpc_delta = [0.0] * self.cfg.horizon
            mpc_x = path_predict[0, :]
            mpc_y = path_predict[1, :]
            mpc_yaw = path_predict[3, :]
            mpc_v = path_predict[2, :]

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict


def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
