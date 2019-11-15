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

    std::string ns = ros::this_node::getNamespace();

    sub_ = nh_.subscribe(in_topic, 0, &RadarObstacleFilter::PointCloudCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      ns + out_topic,1);
  }

  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    double timestamp = msg->header.stamp.toSec();
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::fromROSMsg(*msg, *cloud);

    std::vector<std::vector<std::vector<RadarPoint>>> binned_points;
    binned_points.resize(num_azimuth_bins_);
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      binned_points[i].resize(num_elevation_bins_);
    }

    // iterate over all points in the input cloud
    for (pcl::PointCloud<RadarPoint>::iterator it = cloud->begin(); 
          it != cloud->end(); it++)
    {
      // find azimuth bin index for the current point
      Eigen::Vector2d az_vector(it->x, it->y);
      az_vector.normalize();
      double az_angle = std::copysign(acos(az_vector[0]),az_vector[1]);
      size_t az_idx = size_t(az_angle / bin_width_) + (num_azimuth_bins_ / 2);

      // find elevation bin index for current point
      Eigen::Vector3d el_vector(it->x, it->y, it->z);
      el_vector.normalize();
      double el_angle = asin(el_vector.z());
      size_t el_idx = size_t(el_angle / bin_width_) + (num_elevation_bins_ / 2);

      // find range
      Eigen::Vector3d point(it->x,it->y,it->z);
      double range = point.norm();

      // add to binned points if it is within field of view
      // and its range is within the specified window
      if (std::fabs(az_angle) <= azimuth_fov_
          && std::fabs(el_angle) <= elevation_fov_
          && range <= max_range_
          && range >= min_range_)
      {
        // copy current point
        RadarPoint new_point(*it);

        //LOG(ERROR) << "raw unit vector: " << point.normalized().transpose();
        //LOG(ERROR) << "    discretized: " << ray_table_[az_idx][el_idx].transpose() << "\n\n";

        // adjust xyz to bin center
        new_point.x = ray_table_[az_idx][el_idx].x() * range;
        new_point.y = ray_table_[az_idx][el_idx].y() * range;
        new_point.z = ray_table_[az_idx][el_idx].z() * range;
        new_point.range = range;

        // add to binned points
        binned_points[az_idx][el_idx].push_back(new_point);
      }
    }

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

    // add fake return at max range to each empty bin
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        if (binned_points[i][j].size() == 0)
        {
          RadarPoint fake_point;
          fake_point.range = max_range_;
          fake_point.intensity = 0.0;
          fake_point.doppler = 0.0;
          fake_point.x = ray_table_[i][j].x() * max_range_;
          fake_point.y = ray_table_[i][j].y() * max_range_;
          fake_point.z = ray_table_[i][j].z() * max_range_;
          binned_points[i][j].push_back(fake_point);
        }
      }
    }

    // publish new pointcloud
    pcl::PointCloud<RadarPoint>::Ptr out_cloud(boost::make_shared<pcl::PointCloud<RadarPoint>>());
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        for (int k = 0; k < binned_points[i][j].size(); k++)
        {
          out_cloud->push_back(binned_points[i][j][k]);
        }
      }
    }

    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl::PCLPointCloud2 out_cloud2;
    pcl::toPCLPointCloud2(*out_cloud, out_cloud2);
    pcl_conversions::fromPCL(out_cloud2, out_cloud_msg);
    out_cloud_msg.header.stamp = ros::Time(timestamp);
    out_cloud_msg.header.frame_id = msg->header.frame_id;
    pub_.publish(out_cloud_msg);
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