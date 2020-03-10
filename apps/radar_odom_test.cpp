#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <Map.h>
#include <PoseParameterBlock.h>
#include <PoseParameterization.h>
#include <HomogeneousPointParameterBlock.h>
#include <HomogeneousPointParameterization.h>
#include <PointClusterCostFunction.h>
#include <Transformation.h>
#include <IdProvider.h>
#include <condition_variable>
#include "DataTypes.h"

class ClusterOdometryTester
{
public:

  ClusterOdometryTester(ros::NodeHandle nh) : map_ptr_(new Map())
  {
    nh_ = nh;
    std::string pcl_topic;
    std::string odom_topic;
    nh_.getParam("pcl_topic", pcl_topic);
    nh_.getParam("odom_topic", odom_topic);

    LOG(ERROR) << "odometry topic: " << odom_topic;
    LOG(ERROR) << "pointcloud topic: " << pcl_topic;

    odom_publisher_ = nh_.advertise<nav_msgs::Odometry>("/mmWaveDataHdl/odom", 1);
    scatterer_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/mmWaveDataHdl/odom/scatterers",1);

    odom_sub_ = nh_.subscribe(odom_topic,
                              0,
                              &ClusterOdometryTester::odomCallback,
                              this);
    pcl_sub_ = nh_.subscribe(pcl_topic,
                             0,
                             &ClusterOdometryTester::pclCallback,
                             this);

    window_size_ = 20;

    point_loss_ = new ceres::CauchyLoss(0.1);
    map_ptr_->options.num_threads = 4;
  }

  void odomCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    std::lock_guard<std::mutex> lck(odom_mtx_);
    double timestamp = msg->header.stamp.toSec();
    Eigen::Quaterniond orientation(msg->pose.pose.orientation.w,
                                   msg->pose.pose.orientation.x,
                                   msg->pose.pose.orientation.y,
                                   msg->pose.pose.orientation.z);
    Eigen::Vector3d position(msg->pose.pose.position.x,
                             msg->pose.pose.position.y,
                             msg->pose.pose.position.z);
    Transformation T(position, orientation);
    odom_buffer_.push_front(std::make_pair(timestamp,T));
    cv_.notify_one();
  }

  void pclCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // extract pointcloud from msg
    double timestamp = msg->header.stamp.toSec();
    pcl::PointCloud<RadarPoint>::Ptr cloud_s(new pcl::PointCloud<RadarPoint>);
    pcl::fromROSMsg(*msg, *cloud_s);

    // get current pose guess
    Transformation T_WS_guess;
    if (!poses_.empty())
    {
      // get last optimized pose
      Transformation T_WS_prev = poses_.front()->GetEstimate();
      double timestamp_prev = poses_.front()->GetTimestamp();

      // get relative transform between current and last optimized pose from odometry
      Transformation T_01;
      GetRelativePose(timestamp, timestamp_prev, T_01);

      // get current pose guess
      T_WS_guess = T_WS_prev * T_01;
      //LOG(ERROR) << "T_WS_guess:\n" << T_WS_guess.T() << "\n\n"; 
    }
    else
    {
      GetOdom(timestamp, T_WS_guess);
    }

    // add pose to map
    uint64_t id = IdProvider::instance().NewId();
    poses_.push_front(std::make_shared<PoseParameterBlock>(T_WS_guess, id, timestamp));
    map_ptr_->AddParameterBlock(poses_.front(),Map::Parameterization::Pose);

    // transform pointcloud to world frame
    pcl::PointCloud<RadarPoint>::Ptr cloud_w(new pcl::PointCloud<RadarPoint>);
    Eigen::Quaterniond q_ws = T_WS_guess.q();
    Eigen::Vector3d r_ws = T_WS_guess.r();
    pcl::transformPointCloud(*cloud_s,
                             *cloud_w, 
                             r_ws.cast<float>(), 
                             q_ws.cast<float>(), 
                             true);

    // associate points in current cloud to nearest point scatterers
    double dist_threshold = 0.10;
    std::vector<int> matches;
    for (size_t i = 0; i < cloud_w->size(); i++)
    {
      double min_dist = 1.0;
      size_t scatterer_idx = scatterers_.size();
      for (size_t j = 0; j < scatterers_.size(); j++)
      {
        Eigen::Vector3d point(cloud_w->at(i).x, cloud_w->at(i).y, cloud_w->at(i).z);
        Eigen::Vector3d scatterer = scatterers_[j]->GetEstimate().head(3) 
          / scatterers_[j]->GetEstimate()[3];
        double dist = (point - scatterer).norm();
        double sensor_dist = (scatterer - T_WS_guess.r()).norm();
        if (dist < min_dist && sensor_dist < 6.0 && sensor_dist > 2.0)
        {
          min_dist = dist;
          scatterer_idx = j;
        }
      }
      if (min_dist < dist_threshold)
        matches.push_back(scatterer_idx);
      else
        matches.push_back(-1);
    }

    // create cost functions for each associated point and add to map
    for (size_t i = 0; i < matches.size(); i++)
    {
      Eigen::Vector3d target(cloud_s->at(i).x,
                             cloud_s->at(i).y,
                             cloud_s->at(i).z);

      // add target as new scatterer
      if (matches[i] < 0)
      {
        id = IdProvider::instance().NewId();
        Eigen::Vector4d scatterer(cloud_w->at(i).x,
                                  cloud_w->at(i).y,
                                  cloud_w->at(i).z,
                                  1.0);
        scatterers_.push_back(
          std::make_shared<HomogeneousPointParameterBlock>(scatterer,id));
        map_ptr_->AddParameterBlock(scatterers_.back());
        //matches[i] = scatterers_.size() - 1;
        scatterers_.back()->AddObservation(timestamp);
      }
      else
      {
        double weight = 1.0;
        std::shared_ptr<ceres::CostFunction> cost_func = 
          std::make_shared<PointClusterCostFunction>(target,weight);
        map_ptr_->AddResidualBlock(cost_func,
                                   point_loss_,
                                   poses_.front(),
                                   scatterers_[matches[i]]);
        scatterers_[matches[i]]->AddObservation(timestamp);

        if (scatterers_[matches[i]]->GetObservations().size() > 1
          && !scatterers_[matches[i]]->IsInitialized())
          scatterers_[matches[i]]->SetInitialized(true);
      }
    }

    // solve problem
    map_ptr_->Solve();
    //LOG(ERROR) << map_ptr_->summary.FullReport();
    PublishOdom();
    PublishScatterers();

    // remove old states and residuals
    if (poses_.size() > window_size_)
    {
      // remove residuals
      Map::ResidualBlockCollection res_blks = 
        map_ptr_->GetResidualBlocks(poses_.back()->GetId());

      for (size_t i = 0; i < res_blks.size(); i++)
        map_ptr_->RemoveResidualBlock(res_blks[i].residual_block_id);

      // remove pose
      map_ptr_->RemoveParameterBlock(poses_.back());
      poses_.pop_back();
    }

    // remove old scatterers with only one observation
    for (size_t i = 0; i < scatterers_.size(); i++)
    {
      std::vector<double> observations = scatterers_[i]->GetObservations();
      double time_since_last_obs = timestamp - observations.back();

      Map::ResidualBlockCollection res_blks = 
        map_ptr_->GetResidualBlocks(scatterers_[i]->GetId());

      if ((time_since_last_obs > 0.5 && res_blks.size() < 3))
      {
        for (size_t j = 0; j < res_blks.size(); j++) 
          map_ptr_->RemoveResidualBlock(res_blks[j].residual_block_id);
        map_ptr_->RemoveParameterBlock(scatterers_[i]);
        scatterers_.erase(scatterers_.begin() + i);
        i--;
      }
    }
  }

private:
  // ros-related objects
  ros::NodeHandle nh_;
  ros::Publisher odom_publisher_;
  ros::Publisher scatterer_publisher_;
  ros::Subscriber odom_sub_;
  ros::Subscriber pcl_sub_;

  // solver objects
  std::shared_ptr<Map> map_ptr_;
  ceres::LossFunction *point_loss_;
  size_t window_size_;

  // state and residual containers
  std::vector<std::shared_ptr<HomogeneousPointParameterBlock>> scatterers_;
  std::deque<std::shared_ptr<PoseParameterBlock>> poses_;
  std::deque<std::pair<double,Transformation>> odom_buffer_;
  std::mutex odom_mtx_;
  std::condition_variable cv_;

  void PublishOdom()
  {
    nav_msgs::Odometry odom_out;
    odom_out.header.stamp = ros::Time(poses_.front()->GetTimestamp());
    odom_out.header.frame_id = "map";
    odom_out.child_frame_id = "base_link";
    Eigen::Vector3d position = poses_.front()->GetEstimate().r();
    Eigen::Quaterniond orientation = poses_.front()->GetEstimate().q();
    odom_out.pose.pose.position.x = position.x();
    odom_out.pose.pose.position.y = position.y();
    odom_out.pose.pose.position.z = position.z();
    odom_out.pose.pose.orientation.w = orientation.w();
    odom_out.pose.pose.orientation.x = orientation.x();
    odom_out.pose.pose.orientation.y = orientation.y();
    odom_out.pose.pose.orientation.z = orientation.z();
    odom_publisher_.publish(odom_out);
  }

  void PublishScatterers()
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scatterer_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < scatterers_.size(); i++)
    {
      // get residuals connected with current scatterer
      Map::ResidualBlockCollection res_blks = 
        map_ptr_->GetResidualBlocks(scatterers_[i]->GetId());

      // only display if current scatterer has more than one connected residual
      // (this indicates it is observed more than once)
      if (res_blks.size() > 1)
      {
        Eigen::Vector4d h_pw = scatterers_[i]->GetEstimate();
        Eigen::Vector3d point = h_pw.head(3) / h_pw[3];
        scatterer_cloud->push_back(pcl::PointXYZ(point.x(),point.y(),point.z()));
      }
    }

    // convert to pcl2
    pcl::PCLPointCloud2 scatterer_cloud2;
    pcl::toPCLPointCloud2(*scatterer_cloud, scatterer_cloud2);

    // convert to ros message
    sensor_msgs::PointCloud2 out_cloud;
    pcl_conversions::fromPCL(scatterer_cloud2, out_cloud);
    out_cloud.header.stamp = ros::Time(poses_.front()->GetTimestamp());
    out_cloud.header.frame_id = "map";

    scatterer_publisher_.publish(out_cloud);
  }

  void GetRelativePose(double t1, double t0, Transformation &T_rel)
  {
    Transformation T_WS_1;
    Transformation T_WS_0;
    GetOdom(t1, T_WS_1);
    int idx0 = GetOdom(t0, T_WS_0);
    T_rel = T_WS_0.inverse() * T_WS_1;

    //LOG(ERROR) << "T_WS_1:\n" << T_WS_1.T();

    // erase old odom messages
    std::lock_guard<std::mutex> lk(odom_mtx_);
    int num_to_save = 10; // number of messages to save past T_WS_0
    if (odom_buffer_.size()-idx0 > num_to_save)
      odom_buffer_.erase(odom_buffer_.begin()+idx0+num_to_save, odom_buffer_.end());
  }

  int GetOdom(double t, Transformation &T)
  {
    // wait for up-to-date odom data
    std::unique_lock<std::mutex> lk(odom_mtx_);
    if (!cv_.wait_for(lk,
                      std::chrono::milliseconds(500),
                      [&, this]{return odom_buffer_.front().first > t;}))
    {
      LOG(ERROR) << "waiting for odom measurements has failed";
      return -1;
    }

    size_t odom_idx = 0;
    while (odom_idx < odom_buffer_.size() && odom_buffer_[odom_idx].first > t)
      odom_idx++;

    double t_before = odom_buffer_[odom_idx].first;
    double t_after = odom_buffer_[odom_idx-1].first;
    Transformation T_before = odom_buffer_[odom_idx].second;
    Transformation T_after = odom_buffer_[odom_idx-1].second;

    double c = (t - t_before) / (t_after - t_before);

    Eigen::Vector3d r_before = T_before.r();
    Eigen::Vector3d r_after = T_after.r();
    Eigen::Vector3d r = (1.0-c) * r_before + c * r_after;

    Eigen::Quaterniond q_before = T_before.q();
    Eigen::Quaterniond q_after = T_after.q();
    Eigen::Quaterniond q = q_before.slerp(c, q_after);

    T = Transformation(r,q);

    lk.unlock();
    cv_.notify_one();

    return odom_idx;
  }

};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "radar_odom_test");
  ros::NodeHandle nh("~");
  ClusterOdometryTester* odom_tester = new ClusterOdometryTester(nh);

  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();

  return 0;
}