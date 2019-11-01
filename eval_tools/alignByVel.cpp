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

// pulls measurements from a csv file with one measurement per line: timestamp, vx, vy, vz
// stores measurements in internal datastructure
void getMeasurements(std::vector<std::pair<double,Eigen::Vector3d>> &measurements, 
                     std::string filename)
{
  std::ifstream meas_file(filename);
  std::string line;
  while(std::getline(meas_file, line))
  {
    double vals[4];
    std::istringstream ss(line);
    std::string token;
    int i = 0;
    while (std::getline(ss, token, ','))
    {
      if (i < 4) vals[i] = std::stod(token);
      i++;
    }

    if (i < 4)
      LOG(FATAL) << "wrong number of tokens in line";

    Eigen::Vector3d vel;
    vel << vals[1], vals[2], vals[3];

    measurements.push_back(std::make_pair(vals[0],vel));
  }
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

    for (int i = 0; i < width; i++)
      sum_meas += (it + i)->second;

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
                  std::vector<std::pair<double,Eigen::Vector3d>> &meas1)
{
  // create ceres problem
  ceres::Problem problem;
  ceres::Solver::Options options;

  // create single parameter block
  Eigen::Quaterniond q_vr = Eigen::Quaterniond::UnitRandom();
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

  LOG(ERROR) << "found rotation: \n" << q_vr.toRotationMatrix();
}

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  if (argc != 3)
  {
    LOG(FATAL) << "wrong number of arguments\n" 
               << "argument 1: <filename for vicon measurements> \n"
               << "argument 2: <filename for radar measurements>";
  }
  std::string vicon_filename(argv[1]);
  std::string radar_filename(argv[2]);

  std::vector<std::pair<double,Eigen::Vector3d>> vicon_measurements;
  std::vector<std::pair<double,Eigen::Vector3d>> radar_measurements;

  getMeasurements(vicon_measurements, vicon_filename);
  getMeasurements(radar_measurements, radar_filename);
  
  std::vector<std::pair<double,Eigen::Vector3d>> smooth_vicon;
  smooth_vicon = smoothMeasurements(vicon_measurements);
  
  std::vector<std::pair<double,Eigen::Vector3d>> interp_radar;
  interp_radar = temporalAlign(smooth_vicon, radar_measurements);
  
  findRotation(smooth_vicon, interp_radar);
  
  return 0;
}