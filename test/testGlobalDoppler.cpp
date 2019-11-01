#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <GlobalDopplerCostFunction.h>
#include <QuaternionParameterization.h>
#include <Map.h>
#include <VelocityParameterBlock.h>
#include <DeltaParameterBlock.h>
#include <OrientationParameterBlock.h>
#include <IdProvider.h>
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <random>


const double jacobianTolerance = 1.0e-6;

TEST(googleTests, testGlobalDoppler)
{
  // initialize groundtruth states
  // random orientation and positive x velocity
  Eigen::Quaterniond q_ws = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d v_w(1.0,0.0,0.0);
  Eigen::Vector3d v_s = q_ws.toRotationMatrix().transpose() * v_w;

  // generate doppler readings with additive noise
  double x = 10.0;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> d(0, 0.1);
  std::vector<std::pair<double,Eigen::Vector3d>> targets;
  for (int y = -5; y < 5; y += 1)
  {
    for (int z = -5; z < 5; z += 1)
    {
      Eigen::Vector3d point;
      point << x, y, z;
      Eigen::Vector3d ray = point.normalized();

      Eigen::Vector3d v_target = -1.0 * v_s;

      double doppler = v_target.dot(ray) + d(gen);

      targets.push_back(std::pair<double,Eigen::Vector3d>(doppler, point));
    }
  }

  uint64_t id = IdProvider::instance().NewId();
  std::shared_ptr<OrientationParameterBlock> q_ws_est = 
    std::make_shared<OrientationParameterBlock>(q_ws,id,0.0);
  id = IdProvider::instance().NewId();
  std::shared_ptr<VelocityParameterBlock> v_w_est = 
    std::make_shared<VelocityParameterBlock>(Eigen::Vector3d(0.5,0.2,-0.6),id,0.0);

  // create ceres problem
  std::shared_ptr<Map> map_ptr 
    = std::shared_ptr<Map>(new Map());

  // add (fixed) initial state parameter blocks
  map_ptr->AddParameterBlock(v_w_est);
  map_ptr->AddParameterBlock(q_ws_est, Map::Parameterization::Orientation);
  Eigen::Matrix3d radar_to_imu = Eigen::Matrix3d::Identity();
  
  // create and add residuals and record their ids
  double weight = 1.0 / double(targets.size());
  double sigma_ratio = 1.0;
  for (size_t i = 0; i < targets.size(); i++)
  {
    std::shared_ptr<ceres::CostFunction> v_cost_func = 
      std::make_shared<GlobalDopplerCostFunction>(targets[i].first,
                                                  targets[i].second,
                                                  radar_to_imu,
                                                  weight,
                                                  sigma_ratio);
    id = IdProvider::instance().NewId();
    std::shared_ptr<DeltaParameterBlock> delta = 
      std::make_shared<DeltaParameterBlock>(Eigen::Vector3d(0.0,0.0,0.0),id,0.0);
    map_ptr->AddParameterBlock(delta);

    ceres::ResidualBlockId id = map_ptr->AddResidualBlock(v_cost_func,
                                                          NULL,
                                                          q_ws_est,
                                                          v_w_est,
                                                          delta);
  }

  GlobalDopplerCostFunction* v_cost_func =
    new GlobalDopplerCostFunction(targets[0].first,
                                  targets[0].second,
                                  radar_to_imu,
                                  weight,
                                  sigma_ratio);
  id = IdProvider::instance().NewId();
  std::shared_ptr<DeltaParameterBlock> delta = 
    std::make_shared<DeltaParameterBlock>(Eigen::Vector3d(0.0,0.0,0.0),id,0.0);
    
  // check jacobians by manual inspection
  // automatic checking is not reliable because the information
  // matrix is a function of the states
  double* parameters[3];
  parameters[0] = q_ws_est->GetParameters();
  parameters[1] = v_w_est->GetParameters();
  parameters[2] = delta->GetParameters();

  double* jacobians[3];
  Eigen::Matrix<double,4,4,Eigen::RowMajor> J0; // w.r.t. orientation 
  jacobians[0] = J0.data();
  Eigen::Matrix<double,4,3,Eigen::RowMajor> J1; // w.r.t. velocity 
  jacobians[1] = J1.data();
  Eigen::Matrix<double,4,3,Eigen::RowMajor> J2; // w.r.t. delta 
  jacobians[2] = J2.data();

  double* jacobians_minimal[2];
  Eigen::Matrix<double,4,3,Eigen::RowMajor> J0_min; // w.r.t. orientation 
  jacobians_minimal[0] = J0_min.data();
  Eigen::Matrix<double,4,3,Eigen::RowMajor> J1_min; // w.r.t. velocity 
  jacobians_minimal[1] = J1_min.data();
  Eigen::Matrix<double,4,3,Eigen::RowMajor> J2_min; // w.r.t. delta 
  jacobians_minimal[2] = J2_min.data();

  Eigen::Vector4d residuals;

  v_cost_func->EvaluateWithMinimalJacobians(parameters, 
                                            residuals.data(), 
                                            jacobians,
                                            jacobians_minimal);

  double dx = 1e-6;
  
  Eigen::Matrix<double,4,3> J0_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_0;
    Eigen::Vector4d residuals_p;
    Eigen::Vector4d residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Eigen::Quaterniond q_ws_temp = q_ws_est->GetEstimate();
    q_ws_est->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    q_ws_est->SetEstimate(q_ws_temp); // reset to initial value
    dp_0[i] = -dx;
    q_ws_est->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    q_ws_est->SetEstimate(q_ws_temp); // reset again
    J0_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0_lift;
  q_ws_est->LiftJacobian(parameters[0], J0_lift.data());
  if ((J0 - J0_numDiff * J0_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian 0 does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J0_min
      << '\n' << "\nnum diff J0: \n" << J0_numDiff  << "\n\n";
  }
  
  Eigen::Matrix<double,4,3> J1_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_1;
    Eigen::Vector4d residuals_p;
    Eigen::Vector4d residuals_m;
    dp_1.setZero();
    dp_1[i] = dx;
    Eigen::Vector3d v_w_temp = v_w_est->GetEstimate(); // save initial state
    v_w_est->Plus(parameters[1],dp_1.data(),parameters[1]);
    v_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    v_w_est->SetEstimate(v_w_temp); // reset
    v_w_est->Plus(parameters[1],dp_1.data(),parameters[1]);
    v_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    v_w_est->SetEstimate(v_w_temp);
    LOG(ERROR) << "residuals p: " << residuals_p.transpose();
    LOG(ERROR) << "residuals m: " << residuals_m.transpose();
    J1_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
  }
  
  if ((J1 - J1_numDiff).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided jacobian 1 does not agree with num diff: "
      << "\nuser provided J1: \n" << J1 
      << "\n\nnum diff J1:\n" << J1_numDiff << "\n\n";
  }

  Eigen::Matrix<double,4,3> J2_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_2;
    Eigen::Vector4d residuals_p;
    Eigen::Vector4d residuals_m;
    dp_2.setZero();
    dp_2[i] = dx;
    Eigen::Vector3d delta_temp = delta->GetEstimate(); // save initial state
    delta->Plus(parameters[2],dp_2.data(),parameters[2]);
    v_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    delta->SetEstimate(delta_temp); // reset
    delta->Plus(parameters[2],dp_2.data(),parameters[2]);
    v_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    delta->SetEstimate(delta_temp);
    J2_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
  }
  
  if ((J2 - J2_numDiff).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided jacobian 2 does not agree with num diff: "
      << "\nuser provided J1: \n" << J2 
      << "\n\nnum diff J1:\n" << J2_numDiff << "\n\n";
  }
 
  // solve the problem and save the state estimates
  map_ptr->Solve();

  LOG(ERROR) << map_ptr->summary.FullReport();

  Eigen::Quaterniond q_ws_err = q_ws_est->GetEstimate() * q_ws.inverse();
  Eigen::Vector3d v_w_err = v_w_est->GetEstimate() - v_w;

  double err_lim = 1.0e-1;
  
  ASSERT_TRUE(q_ws_err.coeffs().head(3).norm() < err_lim) << "orientation error of " 
                                        << q_ws_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << q_ws_est->GetEstimate().coeffs().transpose() << '\n'
                                        << "groundtruth: " << q_ws.coeffs().transpose();
  ASSERT_TRUE(v_w_err.norm() < err_lim) << "velocity error of " << v_w_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << v_w_est->GetEstimate().transpose() << '\n'
                                        << "groundtruth: " << v_w.transpose();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}