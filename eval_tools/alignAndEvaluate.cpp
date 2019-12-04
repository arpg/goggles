#include <glog/logging.h>
#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <GlobalVelocityMeasCostFunction.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

// pulls measurements from a csv file with one measurement per line: timestamp, vx, vy, vz
// stores measurements in internal datastructure
void getMeasurements(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &measurements, 
                     std::vector<std::string> topics, 
                     std::string bagfile_name, 
                     bool using_groundtruth)
{
  // open rosbag and find requested topics
  rosbag::Bag bag;
  bag.open(bagfile_name, rosbag::bagmode::Read);
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  // containers for poses, twists, and timestamps
  std::vector<std::vector<double> timestamps; // used for all topics
  std::vector<std::vector<Eigen::Vector3d>> twists; // only used for non-groundtruth topics
  std::vector<std::vector<Eigen::Quaterniond>> orientations; // used for all topics
  std::vector<Eigen::Vector3d> positions; // only used for groundtuth topic

  timestamps.resize(topics.size());
  twist.resize(topics.size());
  orientation.resize(topics.size());

  // iterate through rosbag and transfer messages to their proper containers
  foreach(rosbag::MessageInstance const m, view)
  {
    std::string topic = m.getTopic();
    size_t topic_index = std::difference(topics.begin(), 
                                         find(topics.begin(), 
                                              topics.end(), 
                                              topic));
    if (topic_index >= topics.size())
      LOG(FATAL) << "topic not found";

    double timestamp;

    // if the current message is for the groundtruth topic
    if (using_groundtruth && topic_index == 0)
    {
      // assuming groundtruth message will be poseStamped

      geometry_msgs::PoseStamped::ConstPtr msg
        = m.instantiate<geometry_msgs::PoseStamped>();

      if (msg == NULL)
        LOG(FATAL) << "wrong message type for groundtruth message";

      timestamp = msg->header.stamp.toSec();

      Eigen::Vector3d position(msg->pose.position.x,
                               msg->pose.position.y,
                               msg->pose.position.z);
      Eigen::Quaterniond orientation(msg->pose.orientation.w,
                                     msg->pose.orientation.x,
                                     msg->pose.orientation.y,
                                     msg->pose.orientation.z);
      positions.push_back(position);
      orientations[topic_index].push_back(orientation);
    }
    else
    {
      // assuming all other messages will be odometry messages
      // with twist in the local frame

      nav_msgs::Odometry::ConstPtr msg = m.instantiate<nav_msgs::Odometry>();

      if (msg == NULL)
        LOG(FATAL) << "wrong message type for non-groundtruth message";

      timestamp = msg->header.stamp.toSec();

      Eigen::Vector3d twist(msg->twist.twist.linear.x,
                            msg->twist.twist.linear.y,
                            msg->twist.twist.linear.z);
      Eigen::Quaterniond orientation(msg->pose.pose.orientation.w,
                                     msg->pose.pose.orientation.x,
                                     msg->pose.pose.orientation.y,
                                     msg->pose.pose.orientation.z);
      twists[topic_index].push_back(twist);
      orientations[topic_index].push_back(orientation);
    }

    timestamps[topic_index].push_back(timestamp);
  }
  bag.close();
}

// smooths measurements in the given vector using a boxcar filter
// with a fixed width of 20
std::vector<std::pair<double,Eigen::Vector3d>> 
smoothMeasurements(std::vector<std::pair<double,Eigen::Vector3d>> meas)
{
  std::vector<std::pair<double,Eigen::Vector3d>> result;
  int width = 20;
  int back = width / 2;
  int forward = width - back;

  if (meas.size() <= width)
    LOG(FATAL) << "given vector does not contain enough measurements to smooth";

  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it;
  it = meas.begin() + back;

  while (it != (meas.end() - forward))
  {
    Eigen::Vector3d sum_meas = Eigen::Vector3d::Zero();

    for (std::vector<std::pair<double,Eigen::Vector3d>>::iterator it2 = it - back;
      it2 < it + forward; it2++)
      sum_meas += it2->second;

    Eigen::Vector3d mean_meas = sum_meas / double(width);
    result.push_back(std::make_pair(it->first,mean_meas));
    it++;
  }

  return result;
}

// resamples the measurements in meas1 to temporally align with those in meas0
// returns resampled meas1 vector
std::vector<std::pair<double,Eigen::Vector3d>> 
temporalAlign(std::vector<std::pair<double,Eigen::Vector3d>> &meas0,
                   std::vector<std::pair<double,Eigen::Vector3d>> &meas1)
{
  std::vector<std::pair<double,Eigen::Vector3d>> result;

  // first ensure the timestamp of it1 occurs before it0
  while (meas0.size() > 0 && meas1.front().first > meas0.front().first)
    meas0.erase(meas0.begin());

  // then ensure first two measurements in meas1 bracket first measurement in meas0
  while (meas1.size() > 1 && meas1[1].first < meas0.front().first)
    meas1.erase(meas1.begin());

  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it0 = meas0.begin();
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it1 = meas1.begin();

  
  // finally iterate through both vectors, creating resampled measurements
  // by interpolating those in meas2 to match the timestamps in meas1
  for ( ; it0 != meas0.end(); it0++)
  {
    // get timestamp to interpolate to
    double ts0 = it0->first;

    // find measurements in meas1 bracketing current measurement in meas0
    while ((it1 + 1) != meas1.end() && (it1 + 1)->first < ts0)
      it1++;

    if (it1->first > ts0)
      LOG(FATAL) << "timestamp from meas0 is greater than the timestamp from meas1";
    if ((it1 + 1)->first < ts0)
      LOG(FATAL) << "timestamp from meas0+1 is less than the timestamp from meas1";

    // interpolate measurement
    double ts1_pre = it1->first;
    double ts1_post = (it1 + 1)->first;
    double r = (ts0 - ts1_pre) / (ts1_post - ts1_pre);
    Eigen::Vector3d v_interp = (1.0 - r) * it1->second + r * (it1 + 1)->second;

    // add interpolated measurement to result vector
    result.push_back(std::make_pair(ts0,v_interp));
  }

  if (result.size() != meas0.size())
    LOG(ERROR) << "size of resampled vector not equal to size of source vector";
  
  return result;
}

// uses full batch optimization to find rotation that best aligns the measurements
// in meas1 with those in meas2
void findRotation(std::vector<std::pair<double,Eigen::Vector3d>> &meas0,
  std::vector<std::pair<double,Eigen::Vector3d>> &meas1,
  Eigen::Quaterniond& q_vr)
{
  // create ceres problem
  ceres::Problem problem;
  ceres::Solver::Options options;

  // create single parameter block
  QuaternionParameterization* qp = new QuaternionParameterization();
  problem.AddParameterBlock(q_vr.coeffs().data(),4);
  problem.SetParameterization(q_vr.coeffs().data(), qp);

  ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

  // add all residual blocks
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it0 = meas0.begin();
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it1 = meas1.begin();
  while (it0 != meas0.end() && it1 != meas1.end())
  {
    ceres::CostFunction* cost_func = 
      new GlobalVelocityMeasCostFunction(it0->second, it1->second);

    problem.AddResidualBlock(cost_func, loss, q_vr.coeffs().data());

    it0++;
    it1++;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(ERROR) << summary.FullReport();

  LOG(ERROR) << "found rotation: \n\n" << q_vr.toRotationMatrix() << "\n\n";
}

void printErrors(std::vector<std::pair<double,Eigen::Vector3d>> vicon,
  std::vector<std::pair<double,Eigen::Vector3d>> radar,
  Eigen::Quaterniond& q_vr)
{
  Eigen::Vector3d sum_sq_errs = Eigen::Vector3d::Zero();

  for (int i = 0; i < vicon.size(); i++)
  {
    Eigen::Vector3d tf_radar = q_vr.toRotationMatrix() * radar[i].second;
    //LOG(ERROR) << "radar: " << tf_radar.transpose();
    //LOG(ERROR) << "vicon: " << vicon[i].second.transpose();
    Eigen::Vector3d err = vicon[i].second - tf_radar;
    sum_sq_errs += err.cwiseProduct(err);
  }
  //LOG(ERROR) << "sum_sq_err: " << sum_sq_errs.transpose();

  Eigen::Vector3d rms_err = (sum_sq_errs / vicon.size()).cwiseSqrt();

  LOG(ERROR) << "RMS err: " << rms_err.transpose();
}

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);

  std::string bagfile_name;
  std::vector<std::string> topics;
  bool using_groundtruth;

  if (argc > 3)
  {
    bagfile_name = std::string(argv[1]);
    using_groundtruth = std::string(argv[2]);
    for (int i = 3; i < argc; i++)
        topics.push_back(std::string(argv[i]));
  }
  else
  {
    LOG(FATAL) << "wrong number of arguments\n" 
               << "argument 1: <filename for rosbag> \n"
               << "argument 2: true if using groundtruth, false if not \n"
               << "arguments 3-n: list of topics to compare with groundtruth topic first if present";
  }

  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> measurements;

  getMeasurements(measurements, topics, bagfile_name, using_groundtruth == "true");
  
  std::vector<std::pair<double,Eigen::Vector3d>> smooth_vicon;
  smooth_vicon = smoothMeasurements(vicon_measurements);
  /*
  for (int i = 0; i < smooth_vicon.size(); i ++)
  {
    LOG(ERROR) << "meas " << i << ": " << smooth_vicon[i].second.transpose();
  }
  */
  std::vector<std::pair<double,Eigen::Vector3d>> interp_radar;
  interp_radar = temporalAlign(smooth_vicon, radar_measurements);
  
  Eigen::Quaterniond q_vr = Eigen::Quaterniond::UnitRandom();
  findRotation(smooth_vicon, interp_radar, q_vr);
  
  printErrors(smooth_vicon, interp_radar, q_vr);
  
  return 0;
}