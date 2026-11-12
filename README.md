# f1tenth_controllers

## Purpose

ROS 2 controller nodes for the F1TENTH vehicle. This package contains:

- MPC path tracking
- Simple pure-pursuit path tracking
- A steering-penalty schedule used by the MPC controller

Both controllers consume odometry and a raceline/waypoint CSV, then publish
Ackermann drive commands.

## Use

Build and source the workspace, then run one controller:

~~~bash
ros2 run f1tenth_controllers mpc_controller
ros2 run f1tenth_controllers pure_pursuit_controller
~~~

The nodes require a waypoint_csv parameter. Example:

~~~bash
ros2 run f1tenth_controllers pure_pursuit_controller --ros-args \
  -p waypoint_csv:=/home/nvidia/f1tenth_ws/src/f1tenth_bringup/paths/raceline.csv
~~~

The controller launch integration belongs in f1tenth_bringup or a race-specific
launch package.

## Configuration

Important parameters include:

- waypoint_csv: raceline or waypoint CSV path
- odom_topic: vehicle odometry input, normally /odom
- drive_topic: Ackermann command output
- wheelbase: vehicle wheelbase, currently 0.324 m
- speed_to_erpm_gain: controller speed conversion fallback
- lookahead and smoothing parameters for pure pursuit
- MPC horizon, timestep, steering limits, and penalty settings

The MPC steering penalty schedule is:

~~~text
config/mpc_steer_penalty_schedule.csv
~~~

## Interfaces

Typical inputs:

- /odom
- TF pose information when configured

Typical output:

- drive, using AckermannDriveStamped

The output should normally flow through ackermann_mux before reaching the VESC
driver.

## Theory and further reading

MPC repeatedly predicts vehicle motion over a short horizon and chooses control
inputs subject to steering and speed limits. Pure pursuit selects a lookahead
point and computes a steering command from vehicle geometry. See the
[F1TENTH control lecture notes](https://f1tenth.github.io/website/learn/), the
[ROS 2 Ackermann message definition](https://docs.ros.org/en/humble/p/ackermann_msgs/),
and the original [pure-pursuit paper](https://www.ri.cmu.edu/publications/automatic-steering-methods-for-autonomous-automobile-path-tracking/).
