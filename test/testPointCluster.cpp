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
  // generate fake pose
  Eigen::Quaterniond q_ws = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d t_w(0.0,0.0,0.0);
  Transform T_WS(t_w, q_ws);

  // generate landmark
  Eigen::Vector4d h_pw(5.0,1.0,-0.5);

  // get landmark in sensor frame
  Eigen::Vector4d h_ps = T_WS.inverse().T() * h_pw;

  // generate landmark measurements with random perturbations
  std::vector<Eigen::Vector3d> targets;
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3d perturbation = Eigen::Vector3d::Random() * 1.0e-3;
    Eigen::Vector3d target = h_ps.head(3) + perturbation;
    targets.push_back(target);
  }

  uint64_t id = IdProvider::instance().NewId();
  std::shared_ptr<PoseParameterBlock> T_WS_est = 
    std::make_shared<PoseParameterBlock>(T_WS,id,0.0);
  id = IdProvider::instance().NewId();
  std::shared_ptr<HomogeneousPointParameterBlock> h_pw_est =
    std::make_shared<HomogeneousPointParameterBlock>(h_pw,id);

  std::shared_ptr<Map> map_ptr = std::make_shared<Map>();
  map_ptr->AddParameterBlock(T_WS_est, Map::Parameterization::Pose);
  map_ptr->AddParameterBlock(h_pw_est, Map::Parameterization::HomogeneousPoint);

  double weight = 1.0 / double(targets.size());
  for (size_t i = 0; i < targets.size(); i++)
  {
    std::shared_ptr<ceres::CostFunction> point_cost_func =
      std::make_shared<PointClusterCostFunction>(targets[i], weight);
    ceres::ResidualBlock id = map_ptr->AddResidualBlock(point_cost_func,
                                                        NULL,
                                                        T_WS_est,
                                                        h_pw_est);
  }

  PointClusterCostFunction* p_cost_func = 
    new PointClusterCostFunction(targets[0], weight);

  // check jacobians by finite differences
  double* parameters[2];
  parameters[0] = T_WS_est->GetParameters();
  parameters[1] = h_pw_est->GetParameters();

  double* jacobians[2];
  Eigen::Matrix<double,3,7,Eigen::RowMajor> J0;
  jacobians[0] = J0.data();
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J1;
  jacobians[1] = J1.data();

  double* jacobians_minimal[2];
  Eigen::Matrix<double,3,6,Eigen::RowMajor> J0_min;
  jacobians_minimal[0] = J0_min.data();
  Eigen::Matrix<double,3,3,Eigen::RowMajor> J1_min;
  jacobians_minimal[1] = J1_min.data();

  Eigen::Vector3d residuals;

  p_cost_func->EvaluateWithMinimalJacobians(parameters,
                                            residuals.data(),
                                            jacobians,
                                            jacobians_minimal);

  double dx = 1.0e-6;
  Eigen::Matrix<double,3,6> J0_min_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    // need to fill this in
  }
  Eigen::Matrix<double,6,7,Eigen::RowMajor> J0_lift;
  T_WS_est->LiftJacobian(parameters[0], J0_lift.data());
  if ((J0 - J0_min_numDiff * J0_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian 0 does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J0_min
      << '\n' << "\nnum diff J0: \n" << J0_min_numDiff  << "\n\n";
  }

  Eigen::Matrix<double,3,3> J1_min_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    // need to fill this in
  }
  if ((J1_min - J1_min_numDiff ).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided jacobian 1 does not agree with num diff: "
      << "\nuser provided J1: \n" << J1_min 
      << "\n\nnum diff J1:\n" << J1_min_numDiff << "\n\n";
  }

  map_ptr->Solve();
  LOG(ERROR) << map_ptr->summary.FullReport();

  // still need to evaluate error in pose and homogeneous point
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}