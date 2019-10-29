#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <GlobalVelocityMeasCostFunction.h>
#include <QuaternionParameterization.h>
#include <Eigen/Core>
#include <fstream>


const double jacobianTolerance = 1.0e-6;

TEST(googleTests, testGlobalVelocityMeas)
{
  // initialize groundtruth states
  // random orientation and positive x velocity
  Eigen::Quaterniond q_vr = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d v_w(1.2,0.3,-0.7);
  Eigen::Vector3d v_s = q_vr.toRotationMatrix().transpose() * v_w;

  Eigen::Quaterniond q_vr_est = Eigen::Quaterniond::UnitRandom();

  // create ceres problem
  std::shared_ptr<ceres::Problem> problem 
    = std::shared_ptr<ceres::Problem>(new ceres::Problem());
  ceres::Solver::Options options;
  //options.check_gradients = true;

  // add (fixed) initial state parameter blocks
  QuaternionParameterization* quat_param = new QuaternionParameterization;
  problem->AddParameterBlock(q_vr_est.coeffs().data(), 4);
  problem->SetParameterization(q_vr_est.coeffs().data(), quat_param);
  
  // create cost function
  GlobalVelocityMeasCostFunction* v_cost_func =
    new GlobalVelocityMeasCostFunction(v_s, v_w);

  problem->AddResidualBlock(v_cost_func, NULL, q_vr_est.coeffs().data());

  // check jacobians by manual inspection
  // automatic checking is not reliable because the information
  // matrix is a function of the states
  double* parameters[1];
  parameters[0] = q_vr_est.coeffs().data();

  double* jacobians[1];
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J; // w.r.t. orientation at t0
  jacobians[0] = J.data();

  double* jacobians_minimal[1];
  Eigen::Matrix<double,3,3,Eigen::RowMajor> J_min; // w.r.t. orientation at t0
  jacobians_minimal[0] = J_min.data();

  Eigen::Vector3d residuals;

  v_cost_func->EvaluateWithMinimalJacobians(parameters, 
                                            residuals.data(), 
                                            jacobians,
                                            jacobians_minimal);

  double dx = 1e-6;
  
  Eigen::Matrix3d J_min_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_0;
    Eigen::Vector3d residuals_p;
    Eigen::Vector3d residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Eigen::Quaterniond q_vr_temp = q_vr_est;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
    q_vr_est = q_vr_temp; // reset to initial value
    dp_0[i] = -dx;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
    q_vr_est = q_vr_temp; // reset again
    J_min_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
  quat_param->liftJacobian(parameters[0], J_lift.data());
  if ((J - J_min_numDiff * J_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J_min
      << '\n' << "\nnum diff J0: \n" << J_min_numDiff  << "\n\n";
  }
 
  // solve the problem and save the state estimates
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem.get(), &summary);

  LOG(INFO) << summary.FullReport();
  
  Eigen::Vector3d err = v_w - q_vr_est.toRotationMatrix() * v_s;

  double err_lim = 1.0e-5;
  
  ASSERT_TRUE(err.norm() < err_lim) << "velocity error of " 
                                        << err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << (q_vr_est.toRotationMatrix() * v_s).transpose() << '\n'
                                        << "groundtruth: " << v_w.transpose();
                                        
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}