#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <GlobalDopplerCostFunction.h>
#include <QuaternionParameterization.h>
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


  Eigen::Quaterniond q_ws_est(q_ws);
  Eigen::Vector3d v_w_est(0,0,0);

  // create ceres problem
  std::shared_ptr<ceres::Problem> problem 
    = std::shared_ptr<ceres::Problem>(new ceres::Problem());
  ceres::Solver::Options options;
  //options.check_gradients = true;
  //options.gradient_check_relative_precision = 1.0e-5;

  // add (fixed) initial state parameter blocks
  QuaternionParameterization* quat_param = new QuaternionParameterization;
  problem->AddParameterBlock(v_w_est.data(),3);
  problem->AddParameterBlock(q_ws_est.coeffs().data(),4);
  problem->SetParameterization(q_ws_est.coeffs().data(), quat_param);
  
  // create and add residuals and record their ids
  double weight = 1.0 / double(targets.size());
  for (size_t i = 0; i < targets.size(); i++)
  {
    ceres::CostFunction* v_cost_func = 
      new GlobalDopplerCostFunction(targets[i].first,
                                    targets[i].second,
                                    weight);
    ceres::ResidualBlockId id = problem->AddResidualBlock(v_cost_func,
                                                          NULL,
                                                          q_ws_est.coeffs().data(),
                                                          v_w_est.data());
  }

  GlobalDopplerCostFunction* v_cost_func =
    new GlobalDopplerCostFunction(targets[0].first,
                                  targets[0].second,
                                  weight);
    
  // check jacobians by manual inspection
  // automatic checking is not reliable because the information
  // matrix is a function of the states
  double* parameters[2];
  parameters[0] = q_ws.coeffs().data();
  parameters[1] = v_w.data();

  double* jacobians[2];
  Eigen::Matrix<double,1,4,Eigen::RowMajor> J0; // w.r.t. orientation at t0
  jacobians[0] = J0.data();
  Eigen::Matrix<double,1,3,Eigen::RowMajor> J1; // w.r.t. velocity at t0
  jacobians[1] = J1.data();

  double* jacobians_minimal[2];
  Eigen::Matrix<double,1,3,Eigen::RowMajor> J0_min; // w.r.t. orientation at t0
  jacobians_minimal[0] = J0_min.data();
  Eigen::Matrix<double,1,3,Eigen::RowMajor> J1_min; // w.r.t. velocity at t0
  jacobians_minimal[1] = J1_min.data();

  double residual;

  v_cost_func->EvaluateWithMinimalJacobians(parameters, 
                                            &residual, 
                                            jacobians,
                                            jacobians_minimal);

  double dx = 1e-6;
  
  Eigen::Matrix<double,1,3> J0_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_0;
    double residual_p;
    double residual_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Eigen::Quaterniond q_ws_temp = q_ws;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,&residual_p,NULL);
    q_ws = q_ws_temp; // reset to initial value
    dp_0[i] = -dx;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    v_cost_func->Evaluate(parameters,&residual_m,NULL);
    q_ws = q_ws_temp; // reset again
    J0_numDiff[i] = (residual_p - residual_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0_lift;
  quat_param->liftJacobian(parameters[0], J0_lift.data());
  if ((J0 - J0_numDiff * J0_lift).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided Jacobian 0 does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J0_min
      << '\n' << "\nnum diff J0: \n" << J0_numDiff  << "\n\n";
  }
  
  Eigen::Matrix<double,1,3> J1_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_1;
    double residual_p;
    double residual_m;
    dp_1.setZero();
    dp_1[i] = dx;
    Eigen::Vector3d v_w_temp = v_w; // save initial state
    v_w = v_w + dp_1;
    v_cost_func->Evaluate(parameters,&residual_p,NULL);
    v_w = v_w_temp; // reset
    v_w = v_w - dp_1;
    v_cost_func->Evaluate(parameters,&residual_m,NULL);
    v_w = v_w_temp;
    J1_numDiff[i] = (residual_p - residual_m) / (2.0*dx);
  }
  
  if ((J1 - J1_numDiff).norm() > jacobianTolerance)
  {
    LOG(ERROR) << "User provided jacobian 1 does not agree with num diff: "
      << "\nuser provided J1: \n" << J1 
      << "\n\nnum diff J1:\n" << J1_numDiff << "\n\n";
  }
 
  // solve the problem and save the state estimates
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem.get(), &summary);

  LOG(INFO) << summary.FullReport();

  Eigen::Quaterniond q_ws_err = q_ws_est * q_ws.inverse();
  Eigen::Vector3d v_w_err = v_w_est - v_w;

  double err_lim = 1.0e-1;
  
  ASSERT_TRUE(q_ws_err.coeffs().head(3).norm() < err_lim) << "orientation error of " 
                                        << q_ws_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << q_ws_est.coeffs().transpose() << '\n'
                                        << "groundtruth: " << q_ws.coeffs().transpose();
  ASSERT_TRUE(v_w_err.norm() < err_lim) << "velocity error of " << v_w_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << v_w_est.transpose() << '\n'
                                        << "groundtruth: " << v_w.transpose();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}