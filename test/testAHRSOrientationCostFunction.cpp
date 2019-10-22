#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <AHRSOrientationCostFunction.h>
#include <QuaternionParameterization.h>
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <random>

const double jacobianTolerance = 1.0e-6;

TEST(googleTests, testAHRSOrientationCostFunction)
{
  // Initialize groundtruth states
  Eigen::Quaterniond q_WS_0_gt = Eigen::Quaterniond::UnitRandom();
  Eigen::Quaterniond q_WS_1_gt = Eigen::Quaterniond::UnitRandom();

  LOG(ERROR) << "groundtruth: " << (q_WS_1_gt.inverse() * q_WS_0_gt).coeffs().transpose();

  // get random perturbations
  double scale = 0.01;
  Eigen::Vector3d delta_0 = scale * Eigen::Vector3d::Random();
  Eigen::Vector3d delta_1 = scale * Eigen::Vector3d::Random();

  // add perturbation to groundtruth to get measurements
  QuaternionParameterization* qp = new QuaternionParameterization;
  Eigen::Quaterniond q_WS_0_meas, q_WS_1_meas;
  qp->Plus(q_WS_0_gt.coeffs().data(), delta_0.data(), q_WS_0_meas.coeffs().data());
  qp->Plus(q_WS_1_gt.coeffs().data(), delta_1.data(), q_WS_1_meas.coeffs().data());

  // set up ceres problem
  std::shared_ptr<ceres::Problem> problem = std::make_shared<ceres::Problem>();
  ceres::Solver::Options options;
  options.num_threads = 1;
  ceres::Solver::Summary summary;

  // add parameter blocks
  Eigen::Quaterniond q_WS_0_est = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_WS_1_est = Eigen::Quaterniond::Identity();

  problem->AddParameterBlock(q_WS_0_est.coeffs().data(), 4);
  problem->AddParameterBlock(q_WS_1_est.coeffs().data(), 4);
  problem->SetParameterization(q_WS_0_est.coeffs().data(), qp);
  problem->SetParameterization(q_WS_1_est.coeffs().data(), qp);

  // add residual
  Eigen::Matrix3d AHRS_to_imu = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond delta_q_meas = q_WS_1_meas.inverse() * q_WS_0_meas;
  AHRSOrientationCostFunction *cost_func = new AHRSOrientationCostFunction(
    delta_q_meas, AHRS_to_imu);

  LOG(ERROR) << "perturbed: " << delta_q_meas.coeffs().transpose();

  problem->AddResidualBlock(cost_func,
                            NULL, 
                            q_WS_0_est.coeffs().data(), 
                            q_WS_1_est.coeffs().data());

  // solve
  ceres::Solve(options, problem.get(), &summary);
  LOG(ERROR) << '\n' << summary.FullReport();

  Eigen::Quaterniond delta_q_est = q_WS_1_est.inverse() * q_WS_0_est;
  Eigen::Quaterniond delta_q_gt = q_WS_1_gt.inverse() * q_WS_0_gt;
  Eigen::Vector3d delta_q_err = 2.0 * (delta_q_est.inverse() * delta_q_gt).vec();

  LOG(ERROR) << "q_0: " << q_WS_0_est.coeffs().transpose();
  LOG(ERROR) << "q_1: " << q_WS_1_est.coeffs().transpose();

  double err_tolerance = 1.0e-3;
  ASSERT_TRUE(delta_q_err.norm() < err_tolerance) << "delta orientation error of "
    << delta_q_err.norm() << " is greater than tolerance of " << err_tolerance << '.'
    << "\ndelta_q_est: " << delta_q_est.coeffs().transpose()
    << "\ndelta_q_gt: " << delta_q_gt.coeffs().transpose();

  // manually check jacobians
  // automatic checking is not reliable because the information
  // matrix is a function of the states
  double* parameters[2];
  parameters[0] = q_WS_0_est.coeffs().data();
  parameters[1] = q_WS_1_est.coeffs().data();

  double* jacobians[2];
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0; // w.r.t. orientation at t0
  jacobians[0] = J0.data();
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J1; // w.r.t. orientation at t1
  jacobians[1] = J1.data();

  double* jacobians_minimal[2];
  Eigen::Matrix<double,3,3,Eigen::RowMajor> J0_min; // w.r.t. orientation at t0
  jacobians_minimal[0] = J0_min.data();
  Eigen::Matrix<double,3,3,Eigen::RowMajor> J1_min; // w.r.t. orientation at t1
  jacobians_minimal[1] = J1_min.data();

  Eigen::Vector3d residuals;

  cost_func->EvaluateWithMinimalJacobians(parameters, 
                                          residuals.data(), 
                                          jacobians,
                                          jacobians_minimal);

  double dx = 1e-6;
  
  Eigen::Matrix<double,3,3> J0_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_0;
    Eigen::Vector3d residuals_p;
    Eigen::Vector3d residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Eigen::Quaterniond q_ws_temp = q_WS_0_gt;
    qp->Plus(parameters[0],dp_0.data(),parameters[0]);
    cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    q_WS_0_gt = q_ws_temp; // reset to initial value
    dp_0[i] = -dx;
    qp->Plus(parameters[0],dp_0.data(),parameters[0]);
    cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    q_WS_0_gt = q_ws_temp; // reset again
    J0_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0_lift;
  qp->liftJacobian(parameters[0], J0_lift.data());
  if ((J0 - J0_numDiff * J0_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian 0 does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J0_min
      << '\n' << "\nnum diff J0: \n" << J0_numDiff  << "\n\n";
  }
  Eigen::Matrix<double,3,3> J1_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_1;
    Eigen::Vector3d residuals_p;
    Eigen::Vector3d residuals_m;
    dp_1.setZero();
    dp_1[i] = dx;
    Eigen::Quaterniond q_ws_temp = q_WS_1_gt;
    qp->Plus(parameters[1],dp_1.data(),parameters[1]);
    cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    q_WS_1_gt = q_ws_temp; // reset to initial value
    dp_1[i] = -dx;
    qp->Plus(parameters[1],dp_1.data(),parameters[1]);
    cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    q_WS_1_gt = q_ws_temp; // reset again
    J1_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J1_lift;
  qp->liftJacobian(parameters[1], J1_lift.data());
  if ((J1 - J1_numDiff * J1_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian 1 does not agree with num diff:"
      << '\n' << "user provided J1: \n" << J1_min
      << '\n' << "\nnum diff J1: \n" << J1_numDiff  << "\n\n";
  }
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}