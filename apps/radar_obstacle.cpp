#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/extract_indices.h>
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
  : nh_(nh)
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
    nh_.getParam("bin_width", bin_width_);

    bin_width_ *= M_PI / 180.0;

    std::string ns = ros::this_node::getNamespace();

    sub_ = nh_.subscribe(in_topic, 1, &RadarObstacleFilter::PointCloudCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      ns + out_topic,1);
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
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(20);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    pcl::PointCloud<RadarPoint>::Ptr clustered_cloud(new pcl::PointCloud<RadarPoint>());

    // determine if clusters are roughly on same ray from sensor
    // if so cluster is likely reflections
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
      it != cluster_indices.end(); it++)
    {
      std::vector<int> indices_copy = it->indices;
      bool replace_with_mean = false;
      if (indices_copy.size() >= 3)
      {
        //LOG(ERROR) << "getting rays";
        // get unit ray pointing toward each point in the cluster
        std::vector<Eigen::Vector3d> rays;
        for (std::vector<int>::const_iterator pit = it->indices.begin();
          pit != it->indices.end(); pit++)
        {
          Eigen::Vector3d unit_ray(cloud->points[*pit].x,
                                   cloud->points[*pit].y,
                                   cloud->points[*pit].z);
          unit_ray.normalize();
          rays.push_back(unit_ray);
        }
        //LOG(ERROR) << "getting angles";
        // determine the angle between each pair of points
        Eigen::MatrixXd angles(rays.size(), rays.size());
        angles.setZero();

        for (int i = 0; i < rays.size(); i++)
        {
          for (int j = i+1; j < rays.size(); j++)
          {
            angles(i,j) = std::fabs(acos(rays[i].dot(rays[j])));
            angles(j,i) = angles(i,j);
          }
        }
        //LOG(ERROR) << "checking for outliers";
        // remove outliers
        int cluster_idx = 0;
        for (int i = 0; i < angles.cols(); i++)
        {
          double min_angle = 10.0;
          for (int j = 0; j < angles.rows(); j++)
          {
            if (i != j && angles(i,j) < min_angle) 
              min_angle = angles(i,j);
          }
          if (min_angle > 2.0 * bin_width_)
          {
            //LOG(ERROR) << "removing outlier at " << cluster_idx << " from cluster of size " << indices_copy.size();
            indices_copy.erase(indices_copy.begin() + cluster_idx);
          }
          else
          {
            cluster_idx++;
          }
        }
        replace_with_mean = indices_copy.size() > 3;
      }
      // if max angle is less than bin width, add the 
      // mean of the points to the new cloud
      if (replace_with_mean)
      {
        RadarPoint mean_point;
        mean_point.x = 0.0;
        mean_point.y = 0.0;
        mean_point.z = 0.0;
        mean_point.intensity = 0.0;
        mean_point.doppler = 0.0;
        mean_point.range = 0.0;
        for (std::vector<int>::const_iterator pit = indices_copy.begin();
          pit != indices_copy.end(); pit++)
        {
          mean_point.x += cloud->points[*pit].x;
          mean_point.y += cloud->points[*pit].y;
          mean_point.z += cloud->points[*pit].z;
          mean_point.intensity += cloud->points[*pit].intensity;
          mean_point.doppler += cloud->points[*pit].doppler;
          mean_point.range += cloud->points[*pit].range;
        }
        mean_point.x /= indices_copy.size();
        mean_point.y /= indices_copy.size();
        mean_point.z /= indices_copy.size();
        mean_point.intensity /= indices_copy.size();
        mean_point.doppler /= indices_copy.size();
        mean_point.range /= indices_copy.size();

        clustered_cloud->points.push_back(mean_point);
      }
      else // add the individual points to the cloud
      {
        for (std::vector<int>::const_iterator pit = it->indices.begin();
          pit != it->indices.end(); pit++)
        {
          clustered_cloud->points.push_back(cloud->points[*pit]);
        }
      }
    }


    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl::PCLPointCloud2 out_cloud2;
    pcl::toPCLPointCloud2(*clustered_cloud, out_cloud2);
    pcl_conversions::fromPCL(out_cloud2, out_cloud_msg);
    out_cloud_msg.header.stamp = ros::Time(timestamp);
    out_cloud_msg.header.frame_id = msg->header.frame_id;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    LOG(INFO) << "execution time: " << elapsed.count();

    pub_.publish(out_cloud_msg);
  }

protected:

  double bin_width_; // bin width in radians
  std::deque<pcl::PointCloud<RadarPoint>> cloud_queue_;

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