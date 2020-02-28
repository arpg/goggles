#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <Map.h>
#include <PoseParameterBlock.h>
#include <PoseParameterization.h>
#include <HomogeneousPointParameterBlock.h>
#include <HomogeneousPointParameterization.h>
#include <PointClusterCostFunction.h>
#include <IdProvider.h>
#include "DataTypes.h"

class ClusterOdometryTester
{
public:

  ClusterOdometryTester(ros::NodeHandle nh) : map_ptr_(new Map())
  {

  }
private:
  std::shared_ptr<Map> map_ptr_;
};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "radar_odom_test");
  ros::NodeHandle nh("~");
  ClusterOdometryTester* odom_tester = new ClusterOdometryTester(nh);

  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}