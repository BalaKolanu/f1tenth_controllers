#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Joy
from tf2_ros import Buffer, TransformListener, TransformException


class PurePursuitController(Node):
    """Simple pure-pursuit controller using the new F1TENTH topic interfaces."""

    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic', 'drive')
        self.declare_parameter('waypoint_csv', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_tf_pose_feedback', True)
        self.declare_parameter('lookahead_distance_m', 0.8)
        self.declare_parameter('wheelbase_m', 0.324)
        self.declare_parameter('target_speed_mps', 1.0)
        self.declare_parameter('max_steer_rad', 0.4189)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('reverse_path', False)
        self.declare_parameter('deadman_button_index', 5)
        self.declare_parameter('joy_topic', '/joy')

        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.drive_topic = str(self.get_parameter('drive_topic').value)
        self.waypoint_csv = str(self.get_parameter('waypoint_csv').value).strip()
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.use_tf_pose_feedback = bool(self.get_parameter('use_tf_pose_feedback').value)
        self.lookahead_distance = max(
            0.05, float(self.get_parameter('lookahead_distance_m').value)
        )
        self.wheelbase = max(1e-3, float(self.get_parameter('wheelbase_m').value))
        self.target_speed = float(self.get_parameter('target_speed_mps').value)
        self.max_steer = abs(float(self.get_parameter('max_steer_rad').value))
        self.control_rate = max(1.0, float(self.get_parameter('control_rate_hz').value))
        self.reverse_path = bool(self.get_parameter('reverse_path').value)
        self.deadman_button_index = int(self.get_parameter('deadman_button_index').value)
        self.joy_topic = str(self.get_parameter('joy_topic').value)

        self.path_xy, self.path_speed = self._load_path(self.waypoint_csv)
        self.last_odom = None
        self.last_joy = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)
        self.create_timer(1.0 / self.control_rate, self._control_cb)

        if self.path_xy is None:
            self.get_logger().error(
                'No waypoint CSV configured. Set waypoint_csv to a raceline CSV before launching.'
            )
        else:
            self.get_logger().info(
                'Pure pursuit ready: %d waypoints, lookahead=%.2f m, speed=%.2f m/s'
                % (len(self.path_xy), self.lookahead_distance, self.target_speed)
            )

    def _load_path(self, filename):
        if not filename:
            return None, None
        path = Path(filename)
        if not path.is_file():
            self.get_logger().error('Waypoint CSV not found: %s' % filename)
            return None, None

        rows = []
        with path.open('r', newline='') as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    values = [float(value.strip()) for value in next(csv.reader([line]))]
                except (ValueError, StopIteration):
                    continue
                if len(values) >= 2:
                    rows.append(values)

        if len(rows) < 3:
            self.get_logger().error('Waypoint CSV needs at least three numeric rows: %s' % filename)
            return None, None

        data = np.asarray(rows, dtype=float)
        xy = data[:, :2]
        if self.reverse_path:
            xy = xy[::-1].copy()

        speed = None
        if data.shape[1] >= 4:
            speed = data[:, 3].copy()
        elif data.shape[1] >= 3:
            speed = data[:, 2].copy()
        if speed is not None and self.reverse_path:
            speed = speed[::-1].copy()
        return xy, speed

    def _odom_cb(self, msg):
        self.last_odom = msg

    def _joy_cb(self, msg):
        self.last_joy = msg

    def _pose(self):
        if self.use_tf_pose_feedback:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rclpy.time.Time()
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                return float(t.x), float(t.y), yaw
            except TransformException:
                pass

        if self.last_odom is None:
            return None
        pose = self.last_odom.pose.pose
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return float(pose.position.x), float(pose.position.y), yaw

    def _control_cb(self):
        if self.path_xy is None:
            self._publish_stop()
            return

        pose = self._pose()
        if pose is None:
            self._publish_stop()
            return

        x, y, yaw = pose
        distances = np.hypot(self.path_xy[:, 0] - x, self.path_xy[:, 1] - y)
        nearest = int(np.argmin(distances))

        target = nearest
        accumulated = 0.0
        while accumulated < self.lookahead_distance:
            next_target = (target + 1) % len(self.path_xy)
            step = float(np.linalg.norm(self.path_xy[next_target] - self.path_xy[target]))
            accumulated += step
            target = next_target
            if target == nearest:
                break

        dx = self.path_xy[target, 0] - x
        dy = self.path_xy[target, 1] - y
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        steering = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            self.lookahead_distance,
        )
        steering = max(-self.max_steer, min(self.max_steer, steering))

        speed = self.target_speed
        if self.path_speed is not None:
            speed = float(self.path_speed[target])
        self._publish_drive(speed, steering)

    def _publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def _publish_stop(self):
        self._publish_drive(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
