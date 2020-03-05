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
  Transformation T_WS(t_w, q_ws);
  int num_lmrks = 10;
  // generate landmarks
  std::vector<Eigen::Vector4d> h_pw;
  std::vector<Eigen::Vector4d> h_ps;
  for (size_t i = 0; i < num_lmrks; i++)
  {
    Eigen::Vector3d point = Eigen::Vector3d::Random() * 5.0;
    h_pw.push_back(Eigen::Vector4d(point[0],point[1],point[2],1.0));
    h_ps.push_back(T_WS.inverse().T() * h_pw[i]);
  }
  

  // generate landmark measurements with random perturbations
  std::vector<Eigen::Vector3d> targets;
  for (size_t i = 0; i < num_lmrks; i++)
  {
    Eigen::Vector3d perturbation = Eigen::Vector3d::Random() * 1.0e-3;
    targets.push_back(h_ps[i].head(3) + perturbation);
  }

  uint64_t id = IdProvider::instance().NewId();
  std::shared_ptr<PoseParameterBlock> T_WS_est = 
    std::make_shared<PoseParameterBlock>(T_WS,id,0.0);
  std::vector<std::shared_ptr<HomogeneousPointParameterBlock>> h_pw_est;
  for (size_t i = 0; i < num_lmrks; i++)
  {
    id = IdProvider::instance().NewId();
    //std::shared_ptr<HomogeneousPointParameterBlock> point = 
    h_pw_est.push_back(std::make_shared<HomogeneousPointParameterBlock>(h_pw[i],id));
  }

  std::shared_ptr<Map> map_ptr = std::make_shared<Map>();
  map_ptr->AddParameterBlock(T_WS_est, Map::Parameterization::Pose);

  for(size_t i = 0; i < num_lmrks; i++)
  {
    map_ptr->AddParameterBlock(h_pw_est[i], Map::Parameterization::HomogeneousPoint);
  }

  double weight = 1.0 / double(targets.size());
  for (size_t i = 0; i < num_lmrks; i++)
  {
    std::shared_ptr<ceres::CostFunction> point_cost_func =
      std::make_shared<PointClusterCostFunction>(targets[i], weight);
    ceres::ResidualBlockId id = map_ptr->AddResidualBlock(point_cost_func,
                                                        NULL,
                                                        T_WS_est,
                                                        h_pw_est[i]);
  }

  PointClusterCostFunction* p_cost_func = 
    new PointClusterCostFunction(targets[0], weight);

  // check jacobians by finite differences
  double* parameters[2];
  parameters[0] = T_WS_est->GetParameters();
  parameters[1] = h_pw_est[0]->GetParameters();

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
  for (size_t i = 0; i < 6; i++)
  {
    Eigen::Matrix<double,6,1> dp_0;
    Eigen::Vector3d residuals_p;
    Eigen::Vector3d residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Transformation T_WS_temp = T_WS_est->GetEstimate();
    T_WS_est->Plus(parameters[0],dp_0.data(),parameters[0]);
    p_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    T_WS_est->SetEstimate(T_WS_temp);
    dp_0[i] = -dx;
    T_WS_est->Plus(parameters[0],dp_0.data(),parameters[0]);
    p_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    T_WS_est->SetEstimate(T_WS_temp);
    J0_min_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
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
    Eigen::Vector3d dp_1;
    Eigen::Vector3d residuals_p;
    Eigen::Vector3d residuals_m;
    dp_1.setZero();
    dp_1[i] = dx;
    Eigen::Vector4d h_pw_temp = h_pw_est[0]->GetEstimate();
    h_pw_est[0]->Plus(parameters[1],dp_1.data(),parameters[1]);
    p_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    h_pw_est[0]->SetEstimate(h_pw_temp);
    dp_1[i] = -dx;
    h_pw_est[0]->Plus(parameters[1],dp_1.data(),parameters[1]);
    p_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    h_pw_est[0]->SetEstimate(h_pw_temp);
    J1_min_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
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