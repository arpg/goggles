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

// detects outliers (large discontinuities) in groundtruth data and deletes them,
// leaves gaps in the groundtrtuh data
void rejectOutliers(std::vector<std::pair<double,Eigen::Vector3d>> &meas)
{
  double threshold = 0.5;
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it_pre0 = meas.begin();
  std::vector<std::pair<double,Eigen::Vector3d>>::iterator it_pre1 = meas.begin() + 1;
  while (it_pre0 != meas.end() && it_pre1 != meas.end())
  {
    if ((it_pre0->second - it_pre1->second).norm() > threshold)
    {
      std::vector<std::pair<double,Eigen::Vector3d>>::iterator it_post0 = it_pre1 + 1;
      std::vector<std::pair<double,Eigen::Vector3d>>::iterator it_post1 = it_post0 + 1;
      while (it_post0 != meas.end() && it_post1 != meas.end() && 
        (it_post0->second - it_post1->second).norm() < threshold)
      {
        it_post0++;
        it_post1++;
      }
      LOG(ERROR) << "found outliers between " << std::fixed << std::setprecision(3) 
                 << it_pre0->first << " and " << it_post1->first;
 
      meas.erase(it_pre0,it_post1 + 1);
      it_pre0 = it_post1;
      it_pre1 = it_post1+1;
    }
    else
    {
      it_pre0++;
      it_pre1++;
    }
  }
}

// pulls measurements from a rosbag for every topic listed in topics vector
void getMeasurements(std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> &measurements, 
                     std::vector<std::string> topics, 
                     size_t num_pose_topics,
                     std::string bagfile_name, 
                     bool using_groundtruth)
{
  // open rosbag and find requested topics
  rosbag::Bag bag;
  bag.open(bagfile_name, rosbag::bagmode::Read);
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  size_t num_topics = topics.size();

  // containers for positions, and timestamps
  // used for pose only topics
  std::vector<std::vector<double>> gt_timestamps;
  gt_timestamps.resize(num_pose_topics);
  std::vector<std::vector<Eigen::Vector3d>> gt_positions;
  gt_positions.resize(num_pose_topics);

  // iterate through rosbag and transfer messages to their proper containers
  foreach(rosbag::MessageInstance const m, view)
  {
    std::string topic = m.getTopic();

    size_t topic_index = std::distance(topics.begin(), 
                                         find(topics.begin(), 
                                              topics.end(), 
                                              topic));

    if (topic_index >= num_topics)
      LOG(FATAL) << "topic not found";

    double timestamp;

    // if the current message is for the groundtruth topic
    if (topic_index < num_pose_topics)
    {
      // assuming groundtruth message will be poseStamped
      geometry_msgs::PoseStamped::ConstPtr pose_msg
        = m.instantiate<geometry_msgs::PoseStamped>();

      if (pose_msg != NULL)
      {
        double timestamp = pose_msg->header.stamp.toSec();
        Eigen::Vector3d position(pose_msg->pose.position.x,
                                 pose_msg->pose.position.y,
                                 pose_msg->pose.position.z);
        gt_positions[topic_index].push_back(position);
        gt_timestamps[topic_index].push_back(timestamp);
      }
      else
      {
        nav_msgs::Odometry::ConstPtr odom_msg
          = m.instantiate<nav_msgs::Odometry>();

        double timestamp = odom_msg->header.stamp.toSec();
        Eigen::Vector3d position(odom_msg->pose.pose.position.x,
                                 odom_msg->pose.pose.position.y,
                                 odom_msg->pose.pose.position.z);
        gt_positions[topic_index].push_back(position);
        gt_timestamps[topic_index].push_back(timestamp);
      }
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

      Eigen::Vector3d twist_g = orientation.toRotationMatrix() * twist_s;
      measurements[topic_index].push_back(std::make_pair(timestamp,twist_g));
    }
  }
  bag.close();

  // if using groundtruth, need to calculate global twist via finite differences
  if (num_pose_topics > 0)
  {
    for (size_t j = 0; j < num_pose_topics; j++)
    {
      for (size_t i = 1; i < gt_positions[j].size() - 1; i++)
      {
        Eigen::Vector3d delta_p = gt_positions[j][i+1] - gt_positions[j][i-1];
        double delta_t = gt_timestamps[j][i+1] - gt_timestamps[j][i-1];
        measurements[j].push_back(std::make_pair(gt_timestamps[j][i], delta_p / delta_t));
      }
      if (using_groundtruth)
        measurements[0] = smoothMeasurements(measurements[0]);
    }
  }

  // check for invalid values
  for (size_t i = 0; i < num_topics; i++)
  {
    std::vector<std::pair<double,Eigen::Vector3d>>::iterator it = measurements[i].begin();
    LOG(ERROR) << "initial size of topic " << topics[i] << " " << measurements[i].size();
    size_t num_erased = 0;
    while (it != measurements[i].end())
    {
      Eigen::Vector3d vel = it->second;
      if (vel.hasNaN())
      {
        num_erased++;
        measurements[i].erase(it);
      }
      else
      {
        it++;
      }
    }
    LOG(ERROR) << "erased " << num_erased << " messages from " << topics[i];
    LOG(ERROR) << "final size of " << measurements[i].size();
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

      // if reference topic has measurements past the end of the current topic
      if ((it1 + 1)->first < ts0)
      {
        continue;
      }
      else
      {
        // interpolate measurement
        double ts1_pre = it1->first;
        double ts1_post = (it1 + 1)->first;
        double r = (ts0 - ts1_pre) / (ts1_post - ts1_pre);
        Eigen::Vector3d v_interp = (1.0 - r) * it1->second + r * (it1 + 1)->second;

        result[i].push_back(std::make_pair(ts0,v_interp));
      }
    }
  }

  // all measurements are aligned to those in index 0
  result[0] = meas[0];

  for (size_t i = 0; i < num_topics; i++)
  {
    while (result[0].size() < result[i].size())
      result[i].erase(result[i].end() - 1);
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
      while (it0->first > it1->first && it1 != meas[i].end())
        it1++;

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
    size_t gt_idx = 0;
    size_t other_idx = 0;
    while (gt_idx < meas[0].size() && other_idx < meas[i].size())
    {
      double timestamp = meas[0][gt_idx].first;
      if (i == 0)
      {
        result[i].push_back(std::make_pair(timestamp,Eigen::Vector3d::Zero()));
        gt_idx++;
      }
      else
      {
        while (gt_idx < meas[0].size() && other_idx < meas[i].size() 
          && timestamp > meas[i][other_idx].first)
        {
          other_idx++;
        }
        Eigen::Vector3d error = meas[i][other_idx].second - meas[0][gt_idx].second;

        if (other_idx > 0 && std::fabs(error.norm() - result[i].back().second.norm()) > 0.25)
        {
          meas[0].erase(meas[0].begin()+gt_idx);
        }
        else
        {
        result[i].push_back(std::make_pair(timestamp,error.cwiseAbs()));
        gt_idx++;
        }
        other_idx++;
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
  size_t num_pose_topics;
  bool using_groundtruth;

  if (argc > 3)
  {
    bagfile_name = std::string(argv[1]);
    using_groundtruth = std::string(argv[2]) == "true";
    num_pose_topics = std::stoi(std::string(argv[3]));
    for (int i = 4; i < argc; i++)
        topics.push_back(std::string(argv[i]));
  }
  else
  {
    LOG(FATAL) << "wrong number of arguments\n" 
               << "argument 1: <filename for rosbag> \n"
               << "argument 2: true if using groundtruth, false if not \n"
               << "argument 3: number of pose-only topics"
               << "arguments 4-n: list of topics to compare with groundtruth topic first if present, pose-only topics first if present";
  }

  std::string name_prefix = bagfile_name.substr(0,bagfile_name.size()-4);

  size_t num_topics = topics.size();
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> measurements;
  measurements.resize(num_topics);

  getMeasurements(measurements,
                  topics, 
                  num_pose_topics, 
                  bagfile_name, 
                  using_groundtruth);
  
  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> aligned_measurements;

  aligned_measurements = temporalAlign(measurements);
  
  if (using_groundtruth)
    rejectOutliers(aligned_measurements[0]);
  
  std::vector<Eigen::Quaterniond> rotations;
  rotations.push_back(Eigen::Quaterniond(1,0,0,0));
  for (int i = 1; i < num_topics; i++)
    rotations.push_back(Eigen::Quaterniond::UnitRandom());
  
  findRotations(aligned_measurements, rotations);

  std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> rotated_measurements;
  rotated_measurements = spatialAlign(aligned_measurements, rotations);
  std::string meas_file_name = name_prefix + "_aligned";
  
  if (using_groundtruth)
  {
    std::vector<std::vector<std::pair<double,Eigen::Vector3d>>> errors;
    errors = getErrors(rotated_measurements);
    std::string err_file_name = name_prefix + "_errors";
    writeToCsv(errors, err_file_name, topics);
  }
  
  writeToCsv(rotated_measurements, meas_file_name, topics);

  return 0;
}