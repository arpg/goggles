Goggles
=======

# 1. Description

ROS package for state estimation using millimeter wave radar data. Includes:
 - radar\_velocity: estimates body-frame velocity of a radar sensor

# 2. Building
The following dependencies need to be installed independently:

## 2.1 Dependencies
 - Ceres Solver

If using TI millimeter wave sensors the ti\_mmwave\_rospkg is also required.

# 2.2 Building

Build using catkin\_make or catkin build

# 3. Use

## 3.1 radar\_velocity

Run using '''rosrun goggles radar\_velocity \_radar\_topic:=\<radar pointcloud topic\>'''
Input pointclouds must have doppler measurements for every point.

If running with TI AWR1843 mmwave sensors you can use the launch file:
'''roslaunch goggles radar\_velocity.launch'''
