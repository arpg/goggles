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
  EIGEN_MAKE_ALIGNED_ALLOCATOR_NEW

  /// \brief Default constructor
  /// \param[in] nh a ros nodehandle
  RadarObstacleFilter(ros::NodeHandle nh) 
  : nh_(nh),
    max_range_(std::numeric_limits<double>::max()), 
    min_range_(0.0)
  {
    bin_width_ = math.pi / 180.0;
    elevation_fov_ = math.pi;
    azimuth_fov_ = math.pi;
    num_elevation_bins_ = elevation_fov_ / bin_width_;
    num_azimuth_bins_ = azimuth_fov_ / bin_width_;
    InitializeNode();
  }

  /** \brief Constructor
    * \param[in] nh a ros nodehandle
    * \param[in] max_range The maximum range to consider
    * \param[in] min_range The minimum range to consider
    * \param[in] bin_width The bin width in radians used for discretization
    * \param[in] num_el_bins The number of bins in the elevation direction
    * \param[in] num_az_bins The number of bins in the azimuth direction
    */
  RadarObstacleFilter(ros::NodeHandle nh, 
                      double max_range, 
                      double min_range,
                      double bin_width,
                      size_t num_el_bins,
                      size_t num_az_bins)
  : nh_(nh),
    max_range_(max_range), 
    min_range_(min_range),
    bin_width_(bin_width),
    num_elevation_bins_(num_el_bins),
    num_azimuth_bins_(num_az_bins_)
  {
    elevation_fov_ = 0.5 * double(num_elevation_bins_) * bin_width_;
    azimuth_fov_ = 0.5 * double(num_azimuth_bins_) * bin_width_;
    InitializeNode();
  }

  /** \brief Constructor
    * \param[in] nh a ros nodehandle
    * \param[in] max_range The maximum range to consider
    * \param[in] min_range The minimum range to consider
    * \param[in] bin_width The bin width in radians used for discretization
    * \param[in] elevation_fov The field of view in elevation in radians
    * \param[in] azimuth_fov The field of view in azimuth in radians
    */
  RadarObstacleFilter(ros::NodeHandle nh, 
                      double max_range, 
                      double min_range,
                      double bin_width,
                      double elevation_fov,
                      double azimuth_fov)
  : nh_(nh),
    max_range_(max_range), 
    min_range_(min_range),
    bin_width_(bin_width),
    elevation_fov_(elevation_fov),
    azimuth_fov_(azimuth_fov)
  {
    num_elevation_bins_ = 2.0 * elevation_fov / bin_width_;
    num_azimuth_bins_ = 2.0 * azimuth_fov / bin_width_;
    InitializeNode();
  } 

  /// \brief Initializes ros node subscriber and publisher
  void InitializeNode()
  {
    std::string in_topic;
    std::string out_topic;
    nh_.getParam("input_cloud", in_topic);
    nh_.getParam("output_cloud", out_topic);

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
    for (pcl::PointCloud<RadarPoint>::iterator it = cloud.begin(); 
          it != cloud.end(); it++)
    {
      // find azimuth bin index for the current point
      Eigen::Vector2d az_vector(it->x, it->y);
      az_vector.normalize();
      double az_angle = math.copy_sign(math.acos(az_vector[0]),az_vector[1]);
      size_t az_idx = size_t(az_angle / bin_width) + (num_azimuth_bins_ / 2);

      // find elevation bin index for current point
      Eigen::Vector2d el_vector(it->x, it->z);
      el_vector.normalize();
      double el_angle = math.copy_sign(math.acos(el_vector[0],el_vector[1]));
      size_t el_idx = size_t(el_angle / bin_width) + (num_elevation_bins_ / 2);

      if (std::fabs(az_angle) <= azimuth_fov_
          && std::fabs(el_angle) <= elevation_fov_)
      {
        // copy current point into the correct bin
        binned_points[az_idx][el_idx].push_back(*it);
      }
    }

    // Sort returns in each bin by range
    for (int i = 0; i < num_azimuth_bins_; i++)
    {
      for (int j = 0; j < num_elevation_bins_; j++)
      {
        std::sort(binned_points[i][j].begin(),
                  binned_points[i][j].end(),
                  /* look up how to specify comparator */);
      }
    }

    // group nearby returns in each bin by intensity

    // add fake return at max range to each empty bin
  }

  
protected:
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