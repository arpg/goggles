#include "glog/logging.h"
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/mlesac.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <BodyDopplerSacProblem.h>

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

TEST(goggleTests, testBodyDopplerSacProblem)
{
  // generate point cloud with small additive noise
  double x = 10.0;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> d(0, 0.05);
  RadarPointCloud targets;
  Eigen::Vector3d v_s_gt(1,0,0);
  int num_inliers = 50;
  for (int i = 0; i < num_inliers; i++)
  {
    RadarPoint point;
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

    targets.push_back(point);
  }

  // add outlier points
  int num_outliers = 25;
  for (int i = 0; i < num_outliers; i++)
  {
    RadarPoint point;
    Eigen::Vector3d eigen_point = Eigen::Vector3d::Random() * 5.0;
    point.x = eigen_point[0];
    point.y = eigen_point[1];
    point.z = eigen_point[2];

    // get random doppler from -5.0 to 5.0
    point.doppler = double(rand() % 100) / 10.0 - 5.0;
    point.range = sqrt(point.x*point.x 
      + point.y*point.y 
      + point.z*point.z);
    point.intensity = 1.0;

    targets.push_back(point);
  }
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}