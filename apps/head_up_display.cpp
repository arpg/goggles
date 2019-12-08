#define PCL_NO_PRECOMPILE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/impl/transforms.hpp>
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
    std::string cam_info_topic = "/camera/camera_info";
    std::string num_scans = "3";

    nh.param("in_image_topic", in_image_topic, in_image_topic);
    nh.param("out_image_topic", out_image_topic, out_image_topic);
    nh.param("in_radar_topic", in_radar_topic, in_radar_topic);
    nh.param("world_frame", world_frame_, world_frame_);
    nh.param("scans_to_display", num_scans, num_scans);
    nh.param("cam_info_topic", cam_info_topic, cam_info_topic);
    scans_to_display_ = std::stoi(num_scans);

    sensor_msgs::ImageConstPtr img = 
      ros::topic::waitForMessage<sensor_msgs::Image>(in_image_topic);
    sensor_msgs::PointCloud2ConstPtr pcl = 
      ros::topic::waitForMessage<sensor_msgs::PointCloud2>(in_radar_topic);
    sensor_msgs::CameraInfoConstPtr cam_info = 
      ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);

    image_frame_ = img->header.frame_id;
    radar_frame_ = pcl->header.frame_id;

    K_ = new cv::Mat(3,3,CV_32F);
    D_ = new cv::Mat(5,1,CV_32F); // assuming plumb-bob for now

    K_->at<double>(0,0) = cam_info->K[0];
    K_->at<double>(0,1) = 0.0;
    K_->at<double>(0,2) = cam_info->K[2];
    K_->at<double>(1,0) = 0.0;
    K_->at<double>(1,1) = cam_info->K[4];
    K_->at<double>(1,2) = cam_info->K[5];
    K_->at<double>(2,0) = 0.0;
    K_->at<double>(2,1) = 0.0;
    K_->at<double>(2,2) = 1.0;

    D_->at<double>(0,0) = cam_info->D[0];
    D_->at<double>(1,0) = cam_info->D[1];
    D_->at<double>(2,0) = cam_info->D[2];
    D_->at<double>(3,1) = cam_info->D[3];
    D_->at<double>(4,1) = cam_info->D[4];

    im_height_ = cam_info->height;
    im_width_ = cam_info->width;

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
    pcl::PointCloud<RadarPoint> cam_frame_scans;
    tf::Transform radar_to_world = world_to_radar.inverse();
    for (size_t i = 0; i < scan_deque_.size(); i++)
    {
      pcl::PointCloud<RadarPoint> cam_frame_cloud;
      tf::Transform cloud_to_cam = radar_to_cam_ * radar_to_world * scan_deque_[i].first;

      pcl_ros::transformPointCloud(scan_deque_[i].second, cam_frame_cloud, cloud_to_cam);  
      cam_frame_scans += cam_frame_cloud;
    }

    // extract image from message
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // convert pcl to inputarray
    std::vector<cv::Point3d> input_points;
    for (size_t i = 0; i < cam_frame_scans.size(); i++)
    {
      cv::Point3d radar_point;
      radar_point.x = cam_frame_scans[i].x;
      radar_point.y  = cam_frame_scans[i].y;
      radar_point.z = cam_frame_scans[i].z;
    }

    // project radar points into the camera
    std::vector<cv::Point2d> projected_points;
    cv::Mat t_vec(3,1,CV_32F);
    t_vec.at<double>(0) = 0.0;
    t_vec.at<double>(1) = 0.0;
    t_vec.at<double>(3) = 0.0;
    cv::Mat r_vec(3,1,CV_32F);
    r_vec.at<double>(0) = 0.0;
    r_vec.at<double>(1) = 0.0;
    r_vec.at<double>(3) = 0.0;
    cv::projectPoints(input_points, r_vec, t_vec, *K_, *D_, projected_points);

    // add radar map to input image
    // if using heat map

    // if just projecting points

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
  size_t im_height_;
  size_t im_width_;
  cv::Mat *D_; // camera distortion parameters
  cv::Mat *K_; // camera intrinsic parameters
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