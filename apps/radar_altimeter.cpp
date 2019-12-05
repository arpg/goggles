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
    nh_.getParam("radar_topic", in_topic);
    nh_.getParam("min_range", min_range_);
    nh_.getParam("max_range", max_range_);

    std::string ns = ros::this_node::getNamespace();

    sub_ = nh_.subscribe(in_topic, 1, &RadarAltimeter::PointCloudCallback, this);

    pub_ = nh_.advertise<sensor_msgs::Range>(
      ns + "mmWaveDataHdl/altitude",1);

    point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      ns + "mmWaveDataHdl/altitude_point",1);
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
    ec.setMinClusterSize(5);
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

    double altitude = max_range_;
    if (cluster_means.size() > 0)
    {
      // find cluster nearest to the sensor
      size_t cluster_idx = 0;
      for (size_t i = 0; i < cluster_means.size(); i++)
      {
        double range = cluster_means[i].norm();
        if (range < altitude) 
        {
          altitude = range;
          cluster_idx = i;
        }
      }
      if (altitude < min_range_)
        altitude = min_range_;

      // publish mean of selected point cluster for debugging purposes
      pcl::PointXYZ out_point(cluster_means[cluster_idx].x(),
                              cluster_means[cluster_idx].y(),
                              cluster_means[cluster_idx].z());
      pcl::PointCloud<pcl::PointXYZ> out_cloud;
      out_cloud.push_back(out_point);
      pcl::PCLPointCloud2 out_cloud2;
      pcl::toPCLPointCloud2(out_cloud, out_cloud2);
      sensor_msgs::PointCloud2 out_cloud_msg;
      pcl_conversions::fromPCL(out_cloud2, out_cloud_msg);
      out_cloud_msg.header = msg->header;
      point_pub_.publish(out_cloud_msg);


      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      LOG(INFO) << "execution time: " << elapsed.count();
    }
    sensor_msgs::Range out_range_msg;
    out_range_msg.header = msg->header;
    out_range_msg.min_range = min_range_;
    out_range_msg.max_range = max_range_;
    out_range_msg.range = altitude;
    pub_.publish(out_range_msg);
  }

protected:

  double min_range_, max_range_;

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Publisher point_pub_;
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