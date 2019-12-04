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

// smooths measurements in the given vector using a boxcar filter
// with a fixed width of 20
std::vector<std::pair<double,Eigen::Vector3d>> 
smoothMeasurements(std::vector<std::pair<double,Eigen::Vector3d>> &meas)
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

  // containers for positions, and timestamps
  // used for groundtruth only
  std::vector<double> gt_timestamps;
  std::vector<Eigen::Vector3d> gt_positions;

  // iterate through rosbag and transfer messages to their proper containers
  foreach(rosbag::MessageInstance const m, view)
  {
    std::string topic = m.getTopic();

    size_t topic_index = std::distance(topics.begin(), 
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

      double timestamp = msg->header.stamp.toSec();

      Eigen::Vector3d position(msg->pose.position.x,
                               msg->pose.position.y,
                               msg->pose.position.z);
      gt_positions.push_back(position);
      gt_timestamps.push_back(timestamp);
    }
    else
    {
      // assuming all other messages will be odometry messages
      // with twist in the local frame
      nav_msgs::Odometry::ConstPtr msg = m.instantiate<nav_msgs::Odometry>();

      if (msg == NULL)
        LOG(FATAL) << "wrong message type for non-groundtruth message";

      double timestamp = msg->header.stamp.toSec();

      Eigen::Vector3d twist_s(msg->twist.twist.linear.x,
                            msg->twist.twist.linear.y,
                            msg->twist.twist.linear.z);
      Eigen::Quaterniond orientation(msg->pose.pose.orientation.w,
                                     msg->pose.pose.orientation.x,
                                     msg->pose.pose.orientation.y,
                                     msg->pose.pose.orientation.z);

      Eigen::Vector3d twist_g = orientation.toRotationMatrix().inverse() * twist_s;
      measurements[topic_index].push_back(std::make_pair(timestamp,twist_g));
    }
  }
  bag.close();

  // if using groundtruth, need to calculate global twist via finite differences
  if (using_groundtruth)
  {
    for (size_t i = 1; i < gt_positions.size() - 1; i++)
    {
      Eigen::Vector3d delta_p = gt_positions[i+1] - gt_positions[i-1];
      double delta_t = gt_timestamps[i+1] - gt_timestamps[i-1];
      measurements[0].push_back(std::make_pair(gt_timestamps[i], delta_p / delta_t));
    }
    measurements[0] = smoothMeasurements(measurements[0]);
  }
}

// resamples the measurements in meas to temporally align with the measurements
// in the first index of meas
std::vector<std::vector<std::pair<double,Eigen::Vector3d>> >
temporalAlign(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &meas)
{
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> result;
  size_t num_topics = meas.size();
  result.resize(num_topics);

  // ensure measurements in indices 1 through n have timestamps before those in index 0
  for (size_t i = 1; i < num_topics; i++)
  {
    while (meas[0].size() > 0 && meas[i].front().first >= meas[0].front().first)
      meas[0].erase(meas[0].begin());
  }

  // all measurements are aligned to those in index 0
  result[0] = meas[0];

  // align measurements for each topic to those in index 0
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it0; 
  for (size_t i = 1; i < num_topics; i++)
  {
    it0 = meas[0].begin();
    std::vector<std::pair<double,Eigen::Vector3d>>::iterator it1 = meas[i].begin();

    for ( ; it0 != meas[0].end(); it0++)
    {
      // get reference timestamp
      double ts0 = it0->first;

      // find measurements bracketing reference timestamp
      while ((it1 + 1) != meas[i].end() && (it1 + 1)->first < ts0)
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

      result[i].push_back(std::make_pair(ts0,v_interp));
    }

    if (result[i].size() != result[0].size())
      LOG(ERROR) << "size of resampled vector " << i << " not equal to reference vector";
  }
  
  return result;
}

// uses full batch optimization to find rotation that best aligns the measurements
// in meas1 with those in meas2
void findRotations(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>>& meas,
                   std::vector<Eigen::Quaterniond>& rotations)
{
  size_t num_topics = meas.size();
  QuaternionParameterization* qp = new QuaternionParameterization();
  ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

  ceres::Problem::Options prob_options;
  prob_options.local_parameterization_ownership =
    ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  prob_options.loss_function_ownership = 
    ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  prob_options.cost_function_ownership = 
    ceres::Ownership::TAKE_OWNERSHIP;

  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it0;

  for (size_t i = 1; i < num_topics; i++)
  {
    // create problem
    ceres::Problem problem(prob_options);
    ceres::Solver::Options options;

    // add parameter block
    problem.AddParameterBlock(rotations[i].coeffs().data(),4);
    problem.SetParameterization(rotations[i].coeffs().data(), qp);

    // add residual blocks
    std::vector<std::pair<double,Eigen::Vector3d>>::iterator it0 = meas[0].begin();
    std::vector<std::pair<double,Eigen::Vector3d>>::iterator it1 = meas[i].begin();
    while (it0 != meas[0].end() && it1 != meas[i].end())
    {
      ceres::CostFunction* cost_func = 
        new GlobalVelocityMeasCostFunction(it0->second, it1->second);

      problem.AddResidualBlock(cost_func, loss, rotations[i].coeffs().data());

      it0++;
      it1++;
    }

    // solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(ERROR) << "\n\n\nResult for topic " << i << ":\n" << summary.FullReport();
    LOG(ERROR) << "\n\nFound rotation:\n" << rotations[i].toRotationMatrix();
  }

  delete qp;
  delete loss;
}

std::vector<std::vector<std::pair<double,Eigen::Vector3d>>>
spatialAlign(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &meas,
             std::vector<Eigen::Quaterniond> &rotations)
{
  size_t num_topics = rotations.size();
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> result;
  result.resize(num_topics);
  result[0] = meas[0];

  for (size_t i = 1; i < num_topics; i++)
  {
    Eigen::Matrix3d C = rotations[i].toRotationMatrix();
    for (size_t j = 0; j < meas[i].size(); j++)
    {
      double timestamp = meas[i][j].first;
      Eigen::Vector3d rotated_v = C * meas[i][j].second;
      result[i].push_back(std::make_pair(timestamp,rotated_v));
    }
  }
  return result;
}

std::string replaceSlashes(std::string &input)
{
  std::string result = input;
  for (int i = 0; i < input.size(); i++)
  {
    if (result[i] == '/')
      result[i] = '_';
  }
  return result;
}

void writeToCsv(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &meas,
                std::string &filename_prefix,
                std::vector<std::string> &topics)
{
  for (size_t i = 0; i < topics.size(); i++)
  {
    std::string filename = filename_prefix + replaceSlashes(topics[i]) + ".csv";
    std::ofstream out_file(filename, std::ios_base::trunc);

    for (size_t j = 0; j < meas[i].size(); j++)
    {
      out_file << std::fixed << std::setprecision(5) << meas[i][j].first << ","
               << meas[i][j].second.x() << ","
               << meas[i][j].second.y() << ","
               << meas[i][j].second.z() << "\n";
    }
    out_file.close();
  }
}

std::vector<std::vector<std::pair<double,Eigen::Vector3d>>>
getErrors(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &meas)
{
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> result;
  size_t num_topics = meas.size();
  result.resize(num_topics);
  
  for (size_t i = 0; i < num_topics; i++)
  {
    for (size_t j = 0; j < meas[i].size(); j++)
    {
      double timestamp = meas[i][j].first;
      if (i == 0)
      {
        result[i].push_back(std::make_pair(timestamp,Eigen::Vector3d::Zero()));
      }
      else
      {
        Eigen::Vector3d error = meas[i][j].second - meas[0][j].second;
        result[i].push_back(std::make_pair(timestamp,error.cwiseAbs()));
      }
    }
  }

  return result;
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
    using_groundtruth = std::string(argv[2]) == "true";
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

  std::string name_prefix = bagfile_name.substr(0,bagfile_name.size()-4);

  size_t num_topics = topics.size();
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> measurements;
  measurements.resize(num_topics);

  getMeasurements(measurements, topics, bagfile_name, using_groundtruth);
  
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> aligned_measurements;

  aligned_measurements = temporalAlign(measurements);

  std::vector<Eigen::Quaterniond> rotations;
  rotations.push_back(Eigen::Quaterniond(1,0,0,0));
  for (int i = 1; i < num_topics; i++)
    rotations.push_back(Eigen::Quaterniond::UnitRandom());
  
  findRotations(aligned_measurements, rotations);

  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> rotated_measurements;
  rotated_measurements = spatialAlign(aligned_measurements, rotations);
  std::string meas_file_name = name_prefix + "_aligned";
  
  writeToCsv(rotated_measurements, meas_file_name, topics);
  
  if (using_groundtruth)
  {
    std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> errors;
    errors = getErrors(rotated_measurements);
    std::string err_file_name = name_prefix + "_errors";
    writeToCsv(errors, err_file_name, topics);
  }
  
  return 0;
}