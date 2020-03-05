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
#include <Transformation.h>
#include <IdProvider.h>
#include "DataTypes.h"

class ClusterOdometryTester
{
public:

  ClusterOdometryTester(ros::NodeHandle nh) : map_ptr_(new Map())
  {
    nh_ = nh;
    std::string pcl_topic;
    std::string odom_topic;
    nh_.getParam("pcl_topic", pcl_topic);
    nh_.getParam("odom_topic", odom_topic);

    odom_publisher_ = nh_.advertise<nav_msgs::Odometry>("/mmWaveDataHdl/odom", 1);
    lmrk_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/mmWaveDataHdl/odom/landmarks",1);

    odom_sub_ = nh_.subscribe(odom_topic,
                              1,
                              &ClusterOdometryTester::odomCallback,
                              this);
    pcl_sub_ = nh_.subscribe(pcl_topic,
                             1,
                             &ClusterOdometryTester::pclCallback,
                             this);

    window_size_ = 10;

    point_loss_ = new ceres::CauchyLoss(1.0);
    map_ptr_->options.num_threads = 4;
  }

  void odomCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    double timestamp = msg->header.stamp.toSec();
    Eigen::Quaterniond orientation(msg->pose.pose.orientation.w,
                                   msg->pose.pose.orientation.x,
                                   msg->pose.pose.orientation.y,
                                   msg->pose.pose.orientation.z);
    Eigen::Vector3d position(msg->pose.pose.position.x,
                             msg->pose.pose.position.y,
                             msg->pose.pose.position.z);
    Transformation T(position, orientation);
    odom_buffer.push_front(std::make_pair(timestamp,T));
  }

  void pclCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    if (!odom_buffer.empty())
    {
      // extract pointcloud from msg
      double timestamp = msg->header.stamp.toSec();
      pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
      pcl::fromROSMsg(*msg, *cloud);

      // get last optimized pose, if available
      Transformation T_WS_prev;
      if (!poses_.empty())
        T_WS_prev = poses_.front()->GetEstimate();

      // get relative transform between current and last optimized pose

      // get current pose

      // transform pointcloud to world frame

      // associate points in current cloud to nearest point scatterers

      // create cost functions for each associated point and add to map

      // add unassociated points as new scatterers

      // solve problem

      // remove old states and residuals

      // remove old scatterers with only one observation
    }
  }

private:
  // ros-related objects
  ros::NodeHandle nh_;
  ros::Publisher odom_publisher_;
  ros::Publisher lmrk_publisher_;
  ros::Subscriber odom_sub_;
  ros::Subscriber pcl_sub_;

  // solver objects
  std::shared_ptr<Map> map_ptr_;
  ceres::LossFunction *point_loss_;
  size_t window_size_;

  // state and residual containers
  std::vector<std::shared_ptr<HomogeneousPointParameterBlock>> scatterers_;
  std::deque<std::shared_ptr<PoseParameterBlock>> poses_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
  std::deque<std::pair<double,Transformation>> odom_buffer;
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