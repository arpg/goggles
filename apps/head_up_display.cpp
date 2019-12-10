#define PCL_NO_PRECOMPILE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/impl/transforms.hpp>
#include <Eigen/Core>
#include <DataTypes.h>
#include <glog/logging.h>
#include <iostream>
#include <string>
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
    bool use_cam_info;

    nh_.getParam("in_image_topic", in_image_topic);
    nh_.getParam("out_image_topic", out_image_topic);
    nh_.getParam("in_radar_topic", in_radar_topic);
    nh_.getParam("world_frame", world_frame_);
    nh_.getParam("scans_to_display", scans_to_display_);
    nh_.getParam("cam_info_topic", cam_info_topic);
    nh_.getParam("min_range", min_range_);
    nh_.getParam("max_range", max_range_);
    nh_.getParam("use_cam_info", use_cam_info);
    ros::Duration(0.1).sleep();

    sensor_msgs::ImageConstPtr img = 
      ros::topic::waitForMessage<sensor_msgs::Image>(in_image_topic, 
                                                     nh_, 
                                                     ros::Duration(1.0));
    im_height_ = img->height;
    im_width_ = img->width;
    std::string image_frame = img->header.frame_id;
    
    sensor_msgs::PointCloud2ConstPtr pcl =
      ros::topic::waitForMessage<sensor_msgs::PointCloud2>(in_radar_topic, 
                                                           nh_, 
                                                           ros::Duration(1.0));
    char buffer[56];
    size_t length = (pcl->header.frame_id).copy(buffer,56,0);
    buffer[length] = '\0';
    radar_frame_ = std::string(buffer);
    length = (img->header.frame_id).copy(buffer,56,0);
    buffer[length] = '\0';
    image_frame_ = std::string(buffer);

    min_doppler_ = std::numeric_limits<double>::max();
    max_doppler_ = 0.0;

    K_ = std::make_shared<cv::Mat>(3,3,CV_32F);
    D_ = std::make_shared<cv::Mat>(5,1,CV_32F);
    t_ = std::make_shared<cv::Mat>(3,1,CV_32F);
    r_ = std::make_shared<cv::Mat>(3,1,CV_32F);
    if (use_cam_info)
    {
      sensor_msgs::CameraInfoConstPtr cam_info = 
        ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic, 
                                                            nh_, 
                                                            ros::Duration(1.0));

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
    }
    else
    {
      K_->at<float>(0,0) = 284.47;
      K_->at<float>(0,1) = 0.0;
      K_->at<float>(0,2) = 426.27;
      K_->at<float>(1,0) = 0.0;
      K_->at<float>(1,1) = 285.47;
      K_->at<float>(1,2) = 404.12;
      K_->at<float>(2,0) = 0.0;
      K_->at<float>(2,1) = 0.0;
      K_->at<float>(2,2) = 1.0;
      
      D_->at<float>(0) = -0.00272;
      D_->at<float>(1) = 0.03641;
      D_->at<float>(2) = -0.03515;
      D_->at<float>(3) = 0.005939;
      D_->at<float>(4) = 0.0;
    }
    t_->at<float>(0) = 0.0;
    t_->at<float>(1) = 0.0;
    t_->at<float>(3) = 0.0;

    r_->at<float>(0) = 0.0;
    r_->at<float>(1) = 0.0;
    r_->at<float>(3) = 0.0;

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

    radar_sub_ = nh_.subscribe(in_radar_topic, 1, &RadarHud::RadarCallback, this);
    image_sub_ = nh_.subscribe(in_image_topic, 1, &RadarHud::ImageCallback, this);

    image_pub_ = nh_.advertise<sensor_msgs::Image>(out_image_topic,1);
    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("out_cloud",1);
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
    /*
    Eigen::Matrix3d tf_mat;
    tf::matrixTFToEigen(world_to_radar.getBasis(), tf_mat);
    LOG(ERROR) << "radar tf:\n " << tf_mat;
    */
    pcl::PointCloud<RadarPoint> point_cloud;
    pcl::fromROSMsg(*msg, point_cloud);
    scan_deque_.push_front(point_cloud);
    tf_deque_.push_front(world_to_radar);
    if (scan_deque_.size() > scans_to_display_)
    {
      scan_deque_.pop_back();
      tf_deque_.pop_back();
    }
  }

  void GetColor(double input, int &red, int &green, int &blue)
  {
    // calculate color
    double a = (1.0 - input) * 4.0;
    int X = int(a);
    int Y = int(255.0*(a-double(X)));
    switch (X)
    {
    case 0:
      red = 255;
      green = Y;
      blue = 0;
      break;
    case 1:
      red = 255 - Y;
      green = 255;
      blue = 0;
      break;
    case 2:
      red = 0;
      green = 255;
      blue = Y;
      break;
    case 3:
      red = 0;
      green = 255 - Y;
      blue = 255;
      break;
    case 4:
      red = 0; 
      green = 0;
      blue = 255;
      break;
    }
  }

  void AddPointToImg(cv::Mat img, cv::Point2d center, double range, double doppler)
  {
    double radius = (K_->at<float>(0,0) / range) * 0.05;
    int red, green, blue;
    GetColor(doppler, red, green, blue);
    cv::circle(img, 
               center, 
               radius, 
               cv::Scalar(red,green,blue),
               cv::FILLED,
               cv::LINE_8);
  }

  void DrawLegend(cv::Mat img, cv::Point2i coordinate)
  {
    std::stringstream min_ss;
    std::stringstream max_ss;
    min_ss << std::fixed << std::setprecision(2) << min_doppler_;
    max_ss << std::fixed << std::setprecision(2) << max_doppler_;
    std::string min_doppler = min_ss.str();
    std::string max_doppler = max_ss.str();
    
    int segment_x = coordinate.x;
    int segment_y = coordinate.y;
    int baseline;

    cv::Size text_size = cv::getTextSize(min_doppler,
                                         cv::FONT_HERSHEY_PLAIN, 
                                         2.0,
                                         2, 
                                         &baseline);

    int width = 2;
    int height = text_size.height;
    int num_segments = 200;

    cv::putText(img,
                "doppler measurement (m/s):",
                cv::Point2i(segment_x, segment_y - 15),
                cv::FONT_HERSHEY_PLAIN,
                2.0,
                cv::Scalar(255,255,255),
                2);

    cv::putText(img, 
                min_doppler, 
                cv::Point2i(segment_x, segment_y + height), 
                cv::FONT_HERSHEY_PLAIN, 
                2.0, 
                cv::Scalar(255,255,255),
                2);

    segment_x += text_size.width + 5;

    for (int i = 0; i < num_segments; i++)
    {
      double y = double(i) / double(num_segments);
      int red, green, blue;
      GetColor(y, red, green, blue);
      cv::Point2i p1(segment_x, segment_y);
      cv::Point2i p2(segment_x + width, segment_y + height);
      cv::rectangle(img, p1, p2, cv::Scalar(red,green,blue),cv::FILLED,cv::LINE_8);
      segment_x += width;
    }

    segment_x += 5;

    cv::putText(img, 
                max_doppler, 
                cv::Point2i(segment_x, segment_y + height), 
                cv::FONT_HERSHEY_PLAIN, 
                2.0, 
                cv::Scalar(255,255,255),
                2);
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    pcl::PointCloud<RadarPoint> cam_frame_scans;
    pcl::PointCloud<RadarPoint> radar_frame_scans;
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
      for (size_t i = 0; i < scan_deque_.size(); i++)
      {
        pcl::PointCloud<RadarPoint> cam_frame_cloud;
        pcl::PointCloud<RadarPoint> radar_frame_cloud;
        tf::Transform relative_tf = world_to_radar * tf_deque_[i].inverse();
        tf::Transform cloud_to_cam = radar_to_cam_;
        /*
        Eigen::Matrix3d tf_mat;
        tf::matrixTFToEigen(relative_tf.getBasis(), tf_mat);
        LOG(ERROR) << "relative tf:\n " << tf_mat;
        */
        pcl_ros::transformPointCloud(scan_deque_[i], radar_frame_cloud, relative_tf);
        pcl_ros::transformPointCloud(radar_frame_cloud, cam_frame_cloud, cloud_to_cam);  
        cam_frame_scans += cam_frame_cloud;
        radar_frame_scans += radar_frame_cloud;
      }
    }
    if (cam_frame_scans.size() == 0)
      return;

    // apply max and min range check
    pcl::PointCloud<RadarPoint>::iterator it = cam_frame_scans.begin();
    while (it != cam_frame_scans.end())
    {
      Eigen::Vector3d point(it->x, it->y, it->z);
      double range = point.norm();
      if (range > max_range_ || range < min_range_ || point.z() < 0.0)
        cam_frame_scans.erase(it);
      else
        it++;
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
    // and get ranges and doppler measurements
    std::vector<cv::Point3f> input_points;
    std::vector<std::pair<size_t,double>> idx_range_pairs;
    std::vector<double> dopplers;
    
    for (size_t i = 0; i < cam_frame_scans.size(); i++)
    {
      if (cam_frame_scans[i].doppler < min_doppler_)
        min_doppler_ = cam_frame_scans[i].doppler;
      if (cam_frame_scans[i].doppler > max_doppler_)
        max_doppler_ = cam_frame_scans[i].doppler;
    }
    for (size_t i = 0; i < cam_frame_scans.size(); i++)
    {
      cv::Point3f radar_point;
      radar_point.x = float(cam_frame_scans[i].x);
      radar_point.y  = float(cam_frame_scans[i].y);
      radar_point.z = float(cam_frame_scans[i].z);
      input_points.push_back(radar_point);
      double norm_doppler = (cam_frame_scans[i].doppler - min_doppler_) 
                              / (max_doppler_ - min_doppler_);
      dopplers.push_back(norm_doppler);
      idx_range_pairs.push_back(std::make_pair(i,cv::norm(radar_point)));
    }

    // sort by range, longest range first
    std::sort(idx_range_pairs.begin(), 
              idx_range_pairs.end(), 
              [](const std::pair<size_t,double> i, std::pair<size_t,double> j)->bool{return i.second > j.second;});

    // project radar points into the camera
    cv::Mat overlay = cv_ptr->image.clone();
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(input_points, *r_.get(), *t_.get(), *K_.get(), *D_.get(), projected_points);
    for (size_t i = 0; i < projected_points.size(); i++)
    {
      //LOG(ERROR) << "input point: " << input_points[i].x << ", " << input_points[i].y << ", " << input_points[i].z;
      //LOG(ERROR) << "projected point: " << projected_points[i].x << ", " << projected_points[i].y << "\n\n";
      cv::Point2f point = projected_points[idx_range_pairs[i].first];
      if (point.x < im_width_ && point.x >= 0
        && point.y < im_height_ && point.y >=0)
      {
        AddPointToImg(overlay, 
                      point, 
                      idx_range_pairs[i].second,
                      dopplers[idx_range_pairs[i].first]);
      }
    }
    cv::Mat in_img = cv_ptr->image.clone();
    double alpha = 0.375;
    cv::addWeighted(overlay, alpha, in_img, 1.0-alpha, 0, cv_ptr->image);
    /*
    std::string caption = "num radar points: ";
    caption += std::to_string(cam_frame_scans.size());
    cv::putText(cv_ptr->image, 
                caption, 
                cv::Point2f(35, im_height_-30), 
                cv::FONT_HERSHEY_PLAIN, 
                2.0, 
                cv::Scalar(255,255,255),
                2);
    */
    std::string caption = "radar points projected into camera";
    cv::putText(cv_ptr->image, 
                caption, 
                cv::Point2f(35, 40), 
                cv::FONT_HERSHEY_PLAIN, 
                2.0, 
                cv::Scalar(255,255,255),
                2);
    
    DrawLegend(cv_ptr->image, cv::Point2i(35, im_height_-50));

    // publish message
    image_pub_.publish(cv_ptr->toImageMsg());

    pcl::PCLPointCloud2 out_cloud2;
    pcl::toPCLPointCloud2(cam_frame_scans, out_cloud2);
    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl_conversions::fromPCL(out_cloud2, out_cloud_msg);
    out_cloud_msg.header.frame_id = image_frame_;
    pcl_pub_.publish(out_cloud_msg);
  }

protected:
  ros::NodeHandle nh_;
  ros::Publisher image_pub_;
  ros::Publisher pcl_pub_;
  ros::Subscriber radar_sub_;
  ros::Subscriber image_sub_;
  tf::StampedTransform radar_to_cam_;
  tf::TransformListener listener_;
  std::string world_frame_;
  std::mutex scan_mutex_;
  std::string radar_frame_;
  std::string image_frame_;

  int scans_to_display_;
  size_t im_height_;
  size_t im_width_;
  double min_range_;
  double max_range_;
  double max_doppler_;
  double min_doppler_;
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