#define PCL_NO_PRECOMPILE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <DataTypes.h>
#include <glog/logging.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <math.h>
#include <chrono>

class RadarHud
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RadarHud(ros::NodeHandle nh) 
    : image_frame_(""), 
      radar_frame_(""),
      scans_to_display_(3), 
      world_frame_("map")
  {
    nh_ = nh;
    std::string in_image_topic = "/camera/image_raw";
    std::string out_image_topic = "/hud/image";
    std::string in_radar_topic = "/mmWaveDataHdl/RScan";
    std::string num_scans = "3";

    nh.param("in_image_topic", in_image_topic, in_image_topic);
    nh.param("out_image_topic", out_image_topic, out_image_topic);
    nh.param("in_radar_topic", in_radar_topic, in_radar_topic);
    nh.param("world_frame", world_frame_, world_frame_);
    nh.param("scans_to_display", num_scans, num_scans);
    scans_to_display_ = std::stoi(num_scans);

    sensor_msgs::ImageConstPtr img = 
      ros::topic::waitForMessage<sensor_msgs::Image>(in_image_topic);
    sensor_msgs::PointCloud2ConstPtr pcl = 
      ros::topic::waitForMessage<sensor_msgs::PointCloud2>(in_radar_topic);

    image_frame_ = img->header.frame_id;
    radar_frame_ = pcl->header.frame_id;

    listener_.waitForTransform(image_frame_, 
                               radar_frame_, 
                               ros::Time(0), 
                               ros::Duration(1.0));
    listener_.lookupTransform(image_frame_, 
                              radar_frame_, 
                              ros::Time(0), 
                              radar_to_cam_);

    radar_sub_ = nh_.subscribe(in_radar_topic, 1, &RadarHud::RadarCallback, this);
    image_sub_ = nh_.subscribe(in_image_topic, 1, &RadarHud::ImageCallback, this);

    image_pub_ = nh_.advertise<sensor_msgs::Image>(out_image_topic,1);
  }

  void RadarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    tf::StampedTransform world_to_radar;
    listener_.lookupTransform(radar_frame_, 
                              world_frame_, 
                              ros::Time(0), 
                              world_to_radar);
    pcl::PointCloud<RadarPoint> point_cloud;
    pcl::fromROSMsg(*msg, point_cloud);
    scan_deque_.push_front(std::make_pair(world_to_radar,point_cloud));
    if (scan_deque_.size() > scans_to_display_)
      scan_deque_.pop_back();
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    // transform all radar scans to current camera frame
    tf::StampedTransform world_to_radar;
    listener_.lookupTransform(radar_frame_, 
                              world_frame_, 
                              ros::Time(0), 
                              world_to_radar);

    //pcl_ros::transformPointCloud(*raw_cloud, *cloud, radar_to_imu_);

    // extract image from message
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // project radar points into cv mats

    // add radar mats to input image

    // publish message
    image_pub_.publish(cv_ptr->toImageMsg());
  }

protected:
  ros::NodeHandle nh_;
  ros::Publisher image_pub_;
  ros::Subscriber radar_sub_;
  ros::Subscriber image_sub_;
  tf::TransformListener listener_;
  tf::StampedTransform radar_to_cam_;
  std::string image_frame_;
  std::string radar_frame_;
  std::string world_frame_;

  size_t scans_to_display_;
  std::deque<std::pair<tf::StampedTransform,pcl::PointCloud<RadarPoint>>> scan_deque_;
};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "radar_hud");
  ros::NodeHandle nh("~");
  RadarHud* hud = new RadarHud(nh);

  ros::spin();

  return 0;
}