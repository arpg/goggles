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

## 2.2 GTest
 - Do NOT use the package manager to install `libgtest-dev`
 - Install The GTest librairies by following the instructions [here](https://stackoverflow.com/questions/13513905/how-to-set-up-googletest-as-a-shared-library-on-linux)

If using TI millimeter wave sensors the [ti\_mmwave\_rospkg](https://github.com/arpg/ti_mmwave_rospkg) is also required.

## 2.2 Building

Build using catkin\_make or catkin build

# 3. Use

## 3.1 radar\_velocity

Subscribed Topics:<br/>
 - ```<radar pointcloud topic specified by user>```(sensor_msgs/PointCloud2)<br/>
  Input radar measurements, must have doppler field

Published Topics:<br/>
 - ```mmWaveDataHdl/velocity```(geometry_msgs/TwistWithCovarianceStamped)<br/>
Output velocity estimates.

Run using ```rosrun goggles radar_velocity _radar_topic:=<radar pointcloud topic>```

Input pointclouds must have doppler measurements for every point.

The launch file ```radar_velocity.launch``` also launches the TI AWR1843 sensor by default. If you're not using that sensor the parameter ```launch_radar``` can be set to false.

## 3.2 radar\_inertial\_velocity

Subscribed Topics:<br/>
 - ```/imu_data```(sensor_msgs/Imu)<br/>
  Input IMU measurements
  
 - ```/radar_<n>```(sensor_msgs/PointCloud2)<br/>
  Input radar measurements, must have doppler field. Able to accept an arbitrary number of radar input topics. Number of topics is specified in the ```num_radar``` parameter. Radar topic names have the form ```/radar_0```, ```/radar_1```, etc.

Published Topics:<br/>
 - ```mmWaveDataHdl/velocity```(nav_msgs/Odometry)<br/>
Output velocity and orientation estimates.

Parameters:<br/>
 - ```num_radar``` - number of radar boards used 
 - ```config``` - yaml configuration file for the IMU
 - ```publish_imu_state``` - true to publish odom messages at the IMU rate, false to publish at the radar board rate (defaults to false)
 - ```publish_inliers``` - true to publish radar points that pass the sample consensus outlier rejection step (defaults to false)

Run with the following command:<br/>
```rosrun goggles radar_inertial_velocity _num_radar:=<number of radar boards used> _config:=<config file for imu>```

The radar-inertial node requires a config file for the IMU containing priors on the IMU noise and biases. An example config file for the LORD Microstrain 3DM-GX5-25 can be found in the config directory.

Also note the radar-inertial node currently relies on a yaw estimate from the IMU's internal attitude and heading reference system (AHRS). So an IMU with an internal AHRS is required. An option to use magnetometer measurements instead is currently in development.

The radar-inertial node can also be launched with default parameters using the launch file ```radar_inertial_velocity.launch```

As with ```radar_velocity.launch``` the parameter ```launch_radar``` must be set to false if you're not also using the TI AWR1843 board.

## 3.3 radar_altimeter

Measures altitude of a quadrotor using a downward-facing radar sensor. Recommend using 3d sensor with high range resolution and a narrow field of view.

Subscribed Topics:<br/>
 - ```<radar pointcloud topic specified by user>```(sensor_msgs/PointCloud2)<br/>
  Input radar measurements

Published Topics:<br/>
 - ```mmWaveDataHdl/altitude```(sensor_msgs/Range<br/>
Output altitude measurement.

Parameters:<br/>
 - min_range - minimum altitude in meters
 - max_range - maximum altitude in meters
 - frame_id - defaults to ```map```
 - child_frame_id - defaults to ```base_link```

Run with the following command:<br/>
```rosrun goggles radar_altimeter _radar_topic:=<radar pointcloud topic> _min_range:=<min range> _max_range:=<max range>```

## 3.4 head_up_display

Adds radar points colored by their doppler measurement into an image stream. Requires a valid transform from the camera's frame to the radar's frame.

Subscribed Topics:<br/>
 - ```<radar pointcloud topic specified by user>```(sensor_msgs/PointCloud2)<br/>
  Input radar measurements
 - ```<image topic specified by user>``` (sensor_msgs/Image)<br/>
  Input image measurements
 - ```<camera info topic specified by user>``` (sensor_msgs/CameraInfo)<br/>
  Camera calibration

Published Topics:<br/>
 - ```<output image topic specified by user>```(sensor_msgs/Image)<br/>
  Output image.

Parameters:<br/>
 - min_range - minimum range of points to display
 - max_range - maximum range of points to display
 - in_image_topic - topic for raw image
 - out_image_topic - topic for output image with added radar points
 - in_radar_topic - topic for input radar scans
 - scans_to_display - number of past scans to display
 - cam_info_topic - topic for camera calibration parameters

## 3.5 alignAndEvaluate

Found in the ```eval_tools``` directory. Accepts a ros bag with one or more Odometry topic with the estimated velocities of a sensor platform from radar or other means and optionally one Pose topic with the groundtruth poses of the platform from motion capture. ```alignAndEvaluate``` calculates the rotation to align the coordinate frames of the Odometry and Pose topics and outputs the aligned velocity estimates from each topic to csv files. If a groundtruth topic is given, ```alignAndEvaluate``` also outputs the errors between the Odometry topics and groundtruth to csv files.

Command line arguments:
 1) filename for the input ros bag
 2) "true" if groundtruth is provided, "false" if not
 3) a space-separated list of the topics to compare in the input ros bag

 Example command:<br/>
 ```./alignAndEvaluate example.bag true /vrpn_node/Radar/pose /mmWaveDataHdl/velocity /camera/odom/sample```

 Notes:
  - This is not a ROS package. It is built with the ```cmake``` and ```make``` commands.
  - Output files are written to the same directory as the input bag.
  - Output filenames follow the format ```<bagfile name>_aligned_<topic name>.csv``` or ```<bagfile name>_errors_<topic_name>.csv```
