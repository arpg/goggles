#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <PointClusterCostFunction.h>
#include <PoseParameterization.h>
#include <Map.h>
#include <PoseParameterBlock.h>
#include <HomogeneousPointParameterBlock.h>
#include <IdProvider.h>
#include <Eigen/Core>
#include <fstream>
#include <random>

const double jacobianTolerance = 1.0e-6;

TEST(googleTests, testPointCluster)
{

}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}