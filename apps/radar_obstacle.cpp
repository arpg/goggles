#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <DataTypes.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <math.h>
#include <boost/thread.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <chrono>

class RadarObstacleFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Default constructor
  /// \param[in] nh a ros nodehandle
  RadarObstacleFilter(ros::NodeHandle nh) 
  : nh_(nh),
    max_range_(std::numeric_limits<double>::max()), 
    min_range_(0.0)
  {
    InitializeNode();
    MakeRayLookupTable();
  }

  /// \brief Makes a lookup table of rays for each index of elevation and azimuth
  void MakeRayLookupTable()
  {
    std::vector<Eigen::Vector3d> init_list;
    ray_table_.resize(num_azimuth_bins_);
    Eigen::Vector3d init_vec = Eigen::Vector3d::Zero();
    for (int i = 0; i < num_azimuth_bins_; i++)
      ray_table_[i].resize(num_elevation_bins_);

    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        double el_angle = double(j) * bin_width_ - elevation_fov_;
        double az_angle = double(i) * bin_width_ - azimuth_fov_;
        Eigen::Vector3d ray(cos(az_angle) * cos(el_angle),
                            sin(az_angle) * cos(el_angle),
                            sin(el_angle));
        ray_table_[i][j] = ray;
      }
    }
  }

  /// \brief Initializes ros node subscriber and publisher
  void InitializeNode()
  {
    std::string in_topic;
    std::string out_topic;
    nh_.getParam("input_cloud", in_topic);
    nh_.getParam("output_cloud", out_topic);
    nh_.getParam("min_range", min_range_);
    nh_.getParam("max_range", max_range_);
    nh_.getParam("azimuth_fov", azimuth_fov_);
    nh_.getParam("elevation_fov", elevation_fov_);
    nh_.getParam("bin_width", bin_width_);

    // convert degrees to radians
    azimuth_fov_ *= M_PI / 180.0;
    elevation_fov_ *= M_PI / 180.0;
    bin_width_ *= M_PI / 180.0;

    // get num bins
    num_azimuth_bins_ = size_t(2.0 * azimuth_fov_ / bin_width_);
    num_elevation_bins_ = size_t(2.0 * elevation_fov_ / bin_width_);

    binned_points_.resize(num_azimuth_bins_);
    bin_mutexes_.resize(num_azimuth_bins_);
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      binned_points_[i].resize(num_elevation_bins_);
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        bin_mutexes_[i].push_back(new boost::mutex());
      }
    }

    std::string ns = ros::this_node::getNamespace();

    sub_ = nh_.subscribe(in_topic, 1, &RadarObstacleFilter::PointCloudCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      ns + out_topic,1);
  }

  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    auto start = std::chrono::high_resolution_clock::now();
    double timestamp = msg->header.stamp.toSec();
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::fromROSMsg(*msg, *cloud);

    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
        binned_points_[i][j].clear();
    }

    
    boost::asio::io_service io_service;
    boost::thread_group pool;
    boost::asio::io_service::work work(io_service);
    size_t num_threads = 4;

    for (int i = 0; i < num_threads; i++)
      pool.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
    
    // iterate over all points in the input cloud
    for (pcl::PointCloud<RadarPoint>::iterator it = cloud->begin(); 
          it != cloud->end(); it++)
    {
      
      io_service.post(boost::bind(&RadarObstacleFilter::BinRadarReturns,
                                  this,
                                  *it));
      
      //BinRadarReturns(*it);
    }
    io_service.stop();
    pool.join_all();
    /*
    // Sort returns in each bin by range
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        std::sort(binned_points[i][j].begin(),
                  binned_points[i][j].end(),
                  compareRange);
      }
    }

    // group nearby returns in each bin by intensity
    */
    // add fake return at max range to each empty bin
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        if (binned_points_[i][j].size() == 0)
        {
          RadarPoint fake_point;
          fake_point.range = max_range_;
          fake_point.intensity = 0.0;
          fake_point.doppler = 0.0;
          fake_point.x = ray_table_[i][j].x() * max_range_;
          fake_point.y = ray_table_[i][j].y() * max_range_;
          fake_point.z = ray_table_[i][j].z() * max_range_;
          binned_points_[i][j].push_back(fake_point);
        }
      }
    }

    // publish new pointcloud
    pcl::PointCloud<RadarPoint>::Ptr out_cloud(boost::make_shared<pcl::PointCloud<RadarPoint>>());
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        for (int k = 0; k < binned_points_[i][j].size(); k++)
        {
          out_cloud->push_back(binned_points_[i][j][k]);
        }
      }
    }

    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl::PCLPointCloud2 out_cloud2;
    pcl::toPCLPointCloud2(*out_cloud, out_cloud2);
    pcl_conversions::fromPCL(out_cloud2, out_cloud_msg);
    out_cloud_msg.header.stamp = ros::Time(timestamp);
    out_cloud_msg.header.frame_id = msg->header.frame_id;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    LOG(ERROR) << "execution time: " << elapsed.count();

    pub_.publish(out_cloud_msg);
  }

  void BinRadarReturns(RadarPoint& p)
  {
    // find range
    Eigen::Vector3d point(p.x,p.y,p.z);
    double range = point.norm();

    if (range <= max_range_ && range >= min_range_)
    {
      // find azimuth bin index for the current point
      Eigen::Vector2d az_vector(p.x, p.y);
      az_vector.normalize();
      double az_angle = std::copysign(acos(az_vector[0]),az_vector[1]);
      size_t az_idx = size_t(az_angle / bin_width_) + (num_azimuth_bins_ / 2);

      // find elevation bin index for current point
      Eigen::Vector3d el_vector(p.x, p.y, p.z);
      el_vector.normalize();
      double el_angle = asin(el_vector.z());
      size_t el_idx = size_t(el_angle / bin_width_) + (num_elevation_bins_ / 2);

      // add to binned points if it is within field of view
      if (std::fabs(az_angle) <= azimuth_fov_
          && std::fabs(el_angle) <= elevation_fov_)
      {
        // copy current point
        RadarPoint new_point(p);

        // adjust xyz to bin center
        new_point.x = ray_table_[az_idx][el_idx].x() * range;
        new_point.y = ray_table_[az_idx][el_idx].y() * range;
        new_point.z = ray_table_[az_idx][el_idx].z() * range;
        new_point.range = range;

        // add to binned points
        boost::lock_guard<boost::mutex> lck(*bin_mutexes_[az_idx][el_idx]);
        binned_points_[az_idx][el_idx].push_back(new_point);
      }
    }
  }

  
protected:

  struct 
  {
    bool operator()(RadarPoint a, RadarPoint b) const
    {
      return a.range < b.range;
    }
  } compareRange;

  size_t num_elevation_bins_; 
  size_t num_azimuth_bins_;
  double bin_width_; // bin width in radians
  double max_range_;
  double min_range_;
  double azimuth_fov_; // 1/2 azimuth field of view in radians
  double elevation_fov_; // 1/2 elevation field of view in radians

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;

  std::vector<std::vector<Eigen::Vector3d>> ray_table_;
  std::vector<std::vector<std::vector<RadarPoint>>> binned_points_;
  std::vector<std::vector<boost::mutex*>> bin_mutexes_;
};


int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "radar_obstacle");
  ros::NodeHandle nh("~");
  RadarObstacleFilter* obs_filter = new RadarObstacleFilter(nh);

  ros::spin();

  return 0;
}