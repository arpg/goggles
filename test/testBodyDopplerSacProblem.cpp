#define PCL_NO_PRECOMPILE

#include <gtest/gtest.h>
#include "glog/logging.h"
#include <Eigen/Core>
#include <fstream>
#include <random>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/impl/mlesac.hpp>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <BodyDopplerSacModel.h>

struct RadarPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float doppler;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (RadarPoint,
                  (float, x, x)
                  (float, y, y)
                  (float, z, z)
                  (float, intensity, intensity)
                  (float, range, range)
                  (float, doppler, doppler))

typedef pcl::PointCloud<RadarPoint> RadarPointCloud;
typedef pcl::PointCloud<RadarPoint>::Ptr RadarPointCloudPtr;

TEST(goggleTests, testBodyDopplerSacProblem)
{
  // generate point cloud with small additive noise
  double x = 10.0;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> d(0, 0.01);
  RadarPointCloudPtr targets(new RadarPointCloud);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr targets(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int> inliers_gt;

  Eigen::Vector3d v_s_gt(1,0,0);
  int num_inliers = 50;
  for (int i = 0; i < num_inliers; i++)
  {
    RadarPoint point;
    //pcl::PointXYZ point;
    Eigen::Vector3d eigen_point = Eigen::Vector3d::Random() * 5.0;
    point.x = eigen_point[0];
    point.y = eigen_point[1];
    point.z = eigen_point[2];
    Eigen::Vector3d ray = eigen_point.normalized();

    Eigen::Vector3d v_target = -1.0 * v_s_gt;
    
    point.doppler = v_target.dot(ray) + d(gen);
    point.range = sqrt(point.x*point.x 
      + point.y*point.y 
      + point.z*point.z);
    point.intensity = 1.0;
    
    targets->push_back(point);
    inliers_gt.push_back(i);
  }

  // add outlier points
  int num_outliers = 25;
  for (int i = 0; i < num_outliers; i++)
  {
    RadarPoint point;
    //pcl::PointXYZ point;
    Eigen::Vector3d eigen_point = Eigen::Vector3d::Random() * 5.0;
    point.x = eigen_point[0];
    point.y = eigen_point[1];
    point.z = eigen_point[2];
    Eigen::Vector3d ray = eigen_point.normalized();
    Eigen::Vector3d v_target = -1.0 * v_s_gt;

    // get random doppler from -5.0 to 5.0
    point.doppler = double(rand() % 100) / 10.0 - 5.0;

    // ensure random doppler reading is actually an outlier
    double expected_doppler = v_target.dot(ray);
    while (fabs(point.doppler - expected_doppler) < 0.8)
    {
      point.doppler = double(rand() % 100) / 10.0 - 5.0;
    }
    point.range = sqrt(point.x*point.x 
      + point.y*point.y 
      + point.z*point.z);
    point.intensity = 1.0;
    
    targets->push_back(point);
  }

  // create a sample consensus object and compute the model
  pcl::BodyDopplerSacModel<RadarPoint>::Ptr model(
    new pcl::BodyDopplerSacModel<RadarPoint>(targets));
  std::vector<int> inliers;
  pcl::MaximumLikelihoodSampleConsensus<RadarPoint> mlesac(model);
  mlesac.setDistanceThreshold(0.05);
  mlesac.computeModel();
  mlesac.getInliers(inliers);
  Eigen::VectorXf coeffs = mlesac.model_coefficients_;

  // evaluate results 
  LOG(ERROR) << "num inliers: " << inliers.size();
  int wrong_inlier_count = 0;
  for (int i = 0; i < inliers.size(); i++)
  {
    if (inliers[i] > num_inliers)
      wrong_inlier_count++;
  }
  ASSERT_TRUE(wrong_inlier_count == 0) << wrong_inlier_count 
                                       << " outliers wrongly classified as inliers";

  double model_err = (coeffs.cast<double>() - v_s_gt).norm();
  ASSERT_TRUE(model_err < 0.25) << "model coefficients do not match groundtruth \n"
                               << "mlesac model: " << coeffs.transpose() << '\n'
                               << " groundtruth: " << v_s_gt.transpose();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}