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
  QuaternionParameterization* qp = new QuaternionParameterization;
  Eigen::Quaterniond q_WS_gt = Eigen::Quaterniond::UnitRandom();

  // get random perturbations
  double scale = 0.01;
  Eigen::Vector3d delta = scale * Eigen::Vector3d::Random();

  // add perturbation to groundtruth to get measurements
  Eigen::Quaterniond q_WS_meas;
  qp->Plus(q_WS_gt.coeffs().data(), delta.data(), q_WS_meas.coeffs().data());

  // set up ceres problem
  std::shared_ptr<ceres::Problem> problem = std::make_shared<ceres::Problem>();
  ceres::Solver::Options options;
  options.num_threads = 1;
  ceres::Solver::Summary summary;

  // add parameter blocks
  Eigen::Quaterniond q_WS_est = Eigen::Quaterniond::Identity();

  problem->AddParameterBlock(q_WS_est.coeffs().data(), 4);
  problem->SetParameterization(q_WS_est.coeffs().data(), qp);

  // add residual
  Eigen::Matrix3d AHRS_to_imu = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d initial_orientation = Eigen::Matrix3d::Identity();
  AHRSOrientationCostFunction *cost_func = new AHRSOrientationCostFunction(
    q_WS_meas, AHRS_to_imu, initial_orientation, 1.0);

  problem->AddResidualBlock(cost_func,
                            NULL, 
                            q_WS_est.coeffs().data());

  // solve
  ceres::Solve(options, problem.get(), &summary);
  LOG(ERROR) << '\n' << summary.FullReport();


  Eigen::Quaterniond q_err = q_WS_est.inverse() * q_WS_gt;
  
  double err_tolerance = 1.0e-2;
  ASSERT_TRUE(q_err.vec().norm() < err_tolerance) << "delta orientation error of "
    << q_err.vec().norm() << " is greater than tolerance of " << err_tolerance << '.'
    << "\nq_est: " << q_WS_est.coeffs().transpose()
    << "\nq_gt: " << q_WS_gt.coeffs().transpose();
  
  // manually check jacobians
  // automatic checking is not reliable because the information
  // matrix is a function of the states
  double* parameters[1];
  parameters[0] = q_WS_est.coeffs().data();

  double* jacobians[1];
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0; // w.r.t. orientation at t0
  jacobians[0] = J0.data();

  double* jacobians_minimal[1];
  Eigen::Matrix<double,3,3,Eigen::RowMajor> J0_min; // w.r.t. orientation at t0
  jacobians_minimal[0] = J0_min.data();

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
    Eigen::Quaterniond q_ws_temp = q_WS_gt;
    qp->Plus(parameters[0],dp_0.data(),parameters[0]);
    cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    q_WS_gt = q_ws_temp; // reset to initial value
    dp_0[i] = -dx;
    qp->Plus(parameters[0],dp_0.data(),parameters[0]);
    cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    q_WS_gt = q_ws_temp; // reset again
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
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}