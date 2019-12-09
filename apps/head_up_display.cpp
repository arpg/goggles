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

  RadarHud(ros::NodeHandle &nh) 
    : nh_(nh),
      listener_(nh)
  {
    std::string in_image_topic;
    std::string out_image_topic;
    std::string in_radar_topic;
    std::string cam_info_topic;

    nh_.getParam("in_image_topic", in_image_topic);
    nh_.getParam("out_image_topic", out_image_topic);
    nh_.getParam("in_radar_topic", in_radar_topic);
    nh_.getParam("world_frame", world_frame_);
    nh_.getParam("scans_to_display", scans_to_display_);
    nh_.getParam("cam_info_topic", cam_info_topic);
    LOG(ERROR) << "num scan string " << scans_to_display_;
    LOG(ERROR) << "in image topic " << cam_info_topic;
    /*
    sensor_msgs::ImageConstPtr img;
    while (img.get() == NULL) 
    {
      img = ros::topic::waitForMessage<sensor_msgs::Image>(in_image_topic, 
                                                           ros::Duration(10.0));
    }
    sensor_msgs::PointCloud2ConstPtr pcl;
    while (pcl.get() == NULL)
    {
      pcl = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(in_radar_topic, 
                                                                 ros::Duration(10.0));
    }
    //sensor_msgs::CameraInfoConstPtr cam_info = 
    //  ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
    image_frame_ = img->header.frame_id;
    radar_frame_ = pcl->header.frame_id;
    */
    image_frame_ = "/camera_fisheye1_optical_frame";
    radar_frame_ = "/base_radar_link";
    /*
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
    */
    LOG(ERROR) << "setting parameters";
    K_ = std::make_shared<cv::Mat>(3,3,CV_32F);
    K_->at<double>(0,0) = 284.5;
    K_->at<double>(0,1) = 0.0;
    K_->at<double>(0,2) = 421.9;
    K_->at<double>(1,0) = 0.0;
    K_->at<double>(1,1) = 282.9;
    K_->at<double>(1,2) = 398.1;
    K_->at<double>(2,0) = 0.0;
    K_->at<double>(2,1) = 0.0;
    K_->at<double>(2,2) = 1.0;

    D_ = std::make_shared<cv::Mat>(5,1,CV_32F);
    D_->at<double>(0,0) = 0.0;
    D_->at<double>(1,0) = 0.0;
    D_->at<double>(2,0) = 0.0;
    D_->at<double>(3,1) = 0.0;
    D_->at<double>(4,1) = 0.0;

    t_ = std::make_shared<cv::Mat>(3,1,CV_32F);
    t_->at<double>(0) = 0.0;
    t_->at<double>(1) = 0.0;
    t_->at<double>(3) = 0.0;

    r_ = std::make_shared<cv::Mat>(3,1,CV_32F);
    r_->at<double>(0) = 0.0;
    r_->at<double>(1) = 0.0;
    r_->at<double>(3) = 0.0;
    LOG(ERROR) << "waiting for transform from " << image_frame_ << " to " << radar_frame_;
    bool tf_found;
    try
    {
      tf_found = listener_.waitForTransform(image_frame_, 
                               radar_frame_, 
                               ros::Time(0), 
                               ros::Duration(10.0));
    }
    catch (tf::TransformException &e)
    {
      LOG(FATAL) << "waiting for tf failed: " << e.what();
    }
    if (!tf_found)
      LOG(FATAL) << "tf not found";
    LOG(ERROR) << "looking up transform";
    try
    {
      listener_.lookupTransform(image_frame_, 
                                radar_frame_, 
                                ros::Time(0), 
                                radar_to_cam_);
    }
    catch (tf::TransformException &e)
    {
      LOG(FATAL) << "failed to get cam to radar tf: " << e.what();
    }

    LOG(ERROR) << "starting subscribers and publisher";
    radar_sub_ = nh_.subscribe(in_radar_topic, 1, &RadarHud::RadarCallback, this);
    image_sub_ = nh_.subscribe(in_image_topic, 1, &RadarHud::ImageCallback, this);

    image_pub_ = nh_.advertise<sensor_msgs::Image>(out_image_topic,1);
    LOG(ERROR) << "done initializing";
  }

  void RadarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lck(scan_mutex_);
    tf::StampedTransform world_to_radar;
    try
    {
      listener_.lookupTransform(radar_frame_, 
                                world_frame_, 
                                ros::Time(0), 
                                world_to_radar);
    }
    catch (tf::TransformException &e)
    {
      ROS_ERROR("radar transform exception: %s", e.what());
      return;
    }
    pcl::PointCloud<RadarPoint> point_cloud;
    pcl::fromROSMsg(*msg, point_cloud);
    scan_deque_.push_front(point_cloud);
    tf_deque_.push_back(world_to_radar);
    if (scan_deque_.size() > scans_to_display_)
    {
      scan_deque_.pop_back();
      tf_deque_.pop_back();
    }
  }

  void AddPointToImg(cv::Mat img, cv::Point2d center, double weight)
  {
    double radius = weight * 10.0;
    cv::circle( img, center, radius, cv::Scalar(0,0,255),cv::FILLED,cv::LINE_8);
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    pcl::PointCloud<RadarPoint> cam_frame_scans;
    {
      std::lock_guard<std::mutex> lck(scan_mutex_);
      if (scan_deque_.size() == 0)
        return;

      // transform all radar scans to current camera frame
      tf::StampedTransform world_to_radar;
      try
      {
        listener_.lookupTransform(radar_frame_, 
                                  world_frame_, 
                                  ros::Time(0), 
                                  world_to_radar);
      }
      catch (tf::TransformException &e)
      {
        ROS_ERROR("image transform exception: %s", e.what());
        return;
      }

      tf::Transform radar_to_world = world_to_radar.inverse();
      for (size_t i = 0; i < scan_deque_.size(); i++)
      {
        pcl::PointCloud<RadarPoint> cam_frame_cloud;
        tf::Transform cloud_to_cam = radar_to_cam_ * radar_to_world * tf_deque_[i];

        pcl_ros::transformPointCloud(scan_deque_[i], cam_frame_cloud, cloud_to_cam);  
        cam_frame_scans += cam_frame_cloud;
      }
    }
    if (cam_frame_scans.size() == 0)
      return;

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
    // and calculate weights (inverse ranges)
    std::vector<cv::Point3d> input_points;
    std::vector<double> weights;
    for (size_t i = 0; i < cam_frame_scans.size(); i++)
    {
      cv::Point3d radar_point;
      radar_point.x = cam_frame_scans[i].x;
      radar_point.y  = cam_frame_scans[i].y;
      radar_point.z = cam_frame_scans[i].z;
      input_points.push_back(radar_point);
      weights.push_back(1.0 / cv::norm(radar_point));
    }
    // project radar points into the camera
    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(input_points, *r_.get(), *t_.get(), *K_.get(), *D_.get(), projected_points);
    for (size_t i = 0; i < projected_points.size(); i++)
    {
      AddPointToImg(cv_ptr->image, projected_points[i], weights[i]);
    }

    // publish message
    image_pub_.publish(cv_ptr->toImageMsg());
  }

protected:
  ros::NodeHandle nh_;
  ros::Publisher image_pub_;
  ros::Subscriber radar_sub_;
  ros::Subscriber image_sub_;
  tf::StampedTransform radar_to_cam_;
  tf::TransformListener listener_;
  std::string image_frame_;
  std::string radar_frame_;
  std::string world_frame_;
  std::mutex scan_mutex_;

  int scans_to_display_;
  size_t im_height_;
  size_t im_width_;
  std::shared_ptr<cv::Mat> K_; // projection matrix
  std::shared_ptr<cv::Mat> D_; // distortion parameters
  std::shared_ptr<cv::Mat> r_; 
  std::shared_ptr<cv::Mat> t_;
  std::deque<pcl::PointCloud<RadarPoint>> scan_deque_;
  std::deque<tf::StampedTransform> tf_deque_;
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