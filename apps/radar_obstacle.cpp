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
    min_range_(0.0), 
    initialized_(false),
    bin_tolerance_(1.0e-3)
  {
    InitializeNode();
  }

  /// \brief Constructor
  /// \param[in] nh a ros nodehandle
  /// \param[in] max_range The maximum range to consider
  /// \param[in] min_range The minimum range to consider
  RadarObstacleFilter(ros::NodeHandle nh, double max_range, double min_range)
  : nh_(nh),
    max_range_(max_range), 
    min_range_(min_range),
    initialized_(false),
    bin_tolerance_(1.0e-3)
  {
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

    if (!initialized_)
      initialized_ = InitializeFilter(cloud);
  }

  /// \brief Initializes the filter parameters
  /// \param[in] cloud A point cloud to use for the initialization process
  /// \return true if the initialization was successful, false if not
  bool InitializeFilter(const pcl::PointCloud<RadarPoint>::Ptr cloud)
  {
    bool success = false;

    // add points in cloud to elevation and azimuth sets
    for (pcl::PointCloud<RadarPoint>::const_iterator it = cloud->begin();
      it != cloud->end(); it++)
    {
      // create unit vectors in elevation and azimuth from the current point
      Eigen::Vector2d az_vec(it->x, it->y);
      Eigen::Vector2d el_vec(it->x, it->z);
      az_vec.normalize();
      el_vec.normalize();

      // get elevation and azimuth angles
      double azimuth_angle = std::copy_sign(std::acos(az_vec[0]),az_vec[1]);
      double elevation_angle = std::copy_sign(std::acos(el_vec[0]),el_vec[1]);

      // insert new vectors into the elevation and azimuth bin sets
      azimuth_bins_.insert(azimuth_angle);
      elevation_bins_.insert(elevation_angle);
    }

    // detect the elevation and azimuth field of view of the sensor

    // detect the number of elevation and azimuth bins present in the input cloud


    return success;
  }

protected:
  struct AngleCompare
  {
    // compare the input angles within a given tolerance
    bool operator() (const double& lhs, const double& rhs) const
    {
      if (std::fabs(rhs_angle - lhs_angle) < bin_tolerance_)
        return false;
      else
        return lhs_angle < rhs_angle;
    }
  };

  size_t num_az_bins_;
  size_t num_el_bins_;
  double max_range_;
  double min_range_;
  bool initialized_;

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;

  double bin_tolerance_;

  std::set<double> azimuth_bins_;
  std::set<double> elevation_bins_;
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