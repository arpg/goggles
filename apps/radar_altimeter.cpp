#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Range.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Core>
#include <DataTypes.h>
#include <glog/logging.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <math.h>
#include <chrono>

class RadarAltimeter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Default constructor
  /// \param[in] nh a ros nodehandle
  RadarAltimeter(ros::NodeHandle nh) 
  : nh_(nh)
  {
    InitializeNode();
  }

  /// \brief Initializes ros node subscriber and publisher
  void InitializeNode()
  {
    std::string in_topic;
    nh_.getParam("input_cloud", in_topic);
    nh_.getParam("min_range", min_range_);
    nh_.getParam("max_range", max_range_);

    std::string ns = ros::this_node::getNamespace();

    sub_ = nh_.subscribe(in_topic, 1, &RadarAltimeter::PointCloudCallback, this);

    pub_ = nh_.advertise<sensor_msgs::Range>(
      ns + "mmWaveDataHdl/altitude",1);
  }

  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    auto start = std::chrono::high_resolution_clock::now();
    double timestamp = msg->header.stamp.toSec();
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>());
    pcl::fromROSMsg(*msg, *cloud);
    
    // detect point clusters
    pcl::search::KdTree<RadarPoint>::Ptr tree(new pcl::search::KdTree<RadarPoint>());
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<RadarPoint> ec;
    ec.setClusterTolerance(0.2);
    ec.setMinClusterSize(3);
    ec.setMaxClusterSize(20);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // replace clusters with their means
    std::vector<Eigen::Vector3d> cluster_means;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
      it != cluster_indices.end(); it++)
    {
      Eigen::Vector3d sum_cluster = Eigen::Vector3d::Zero();
      for (std::vector<int>::const_iterator pit = it->indices.begin(); 
        pit != it->indices.end(); pit++)
      {
        sum_cluster += Eigen::Vector3d(cloud->points[*pit].x,
                                       cloud->points[*pit].y,
                                       cloud->points[*pit].z);
      }
      cluster_means.push_back(sum_cluster / it->indices.size());
    }

    // find cluster nearest to the sensor
    double altitude = max_range_;
    for (size_t i = 0; i < cluster_means.size(); i++)
    {
      double range = cluster_means[i].norm();
      if (range < altitude) altitude = range;
    }

    sensor_msgs::Range out_range_msg;
    out_range_msg.header = msg->header;
    out_range_msg.min_range = min_range_;
    out_range_msg.max_range = max_range_;
    out_range_msg.range = altitude;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    LOG(INFO) << "execution time: " << elapsed.count();

    pub_.publish(out_range_msg);
  }

protected:

  double min_range_, max_range_;

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
};


int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "radar_altimeter");
  ros::NodeHandle nh("~");
  RadarAltimeter* altimeter = new RadarAltimeter(nh);

  ros::spin();

  return 0;
}