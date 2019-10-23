Goggles
=======

# 1. Description

ROS package for state estimation using millimeter wave radar data. Includes:
 - radar\_velocity: estimates body-frame velocity of a radar sensor
 - radar\_inertial\_velocity: estimates the global frame velocity of a rigidly mounted imu and radar system

# 2. Building
The following dependencies need to be installed independently:

## 2.1 Dependencies
 - [Ceres Solver](http://ceres-solver.org/installation.html)

If using TI millimeter wave sensors the [ti\_mmwave\_rospkg](https://github.com/arpg/ti_mmwave_rospkg) is also required.

## 2.2 Building

Build using catkin\_make or catkin build

# 3. Use

## 3.1 radar\_velocity

Run using ```rosrun goggles radar_velocity _radar_topic:=<radar pointcloud topic>```

Input pointclouds must have doppler measurements for every point.

The launch file ```radar_velocity.launch``` also launches the TI AWR1843 sensor by default. If you're not using that sensor the parameter ```launch_radar``` can be set to false.

## 3.2 radar\_inertial\_velocity

Run with the following command:
```rosrun goggles radar_inertial_velocity _radar_topic:=<radar pointcloud topic> _imu_topic:=<imu topic> _config:=<config file for imu> _imu_ref_frame:=<coordinate frame for the imu> _radar_ref_frame:=<coordinate frame for the radar board>```

The radar-inertial node requires a config file for the IMU containing priors on the IMU noise and biases. An example config file for the LORD Microstrain 3DM-GX5-25 can be found in the config directory.

Also note the radar-inertial node currently relies on a yaw estimate from the IMU's internal attitude and heading reference system (AHRS). So an IMU with an internal AHRS is required. An option to use magnetometer measurements instead is currently in development.

The radar-inertial node can also be launched with default parameters using the launch file ```radar_inertial_velocity.launch```

As with ```radar_velocity.launch``` the parameter ```launch_radar``` must be set to false if you're not also using the TI AWR1843 board.
