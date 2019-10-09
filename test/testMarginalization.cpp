#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <ImuVelocityCostFunction.h>
#include <BodyVelocityCostFunction.h>
#include <MarginalizationError.h>
#include <QuaternionParameterization.h>
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <random>

double sinc_test(double x)
{
  if(fabs(x)>1e-10) 
  {
    return sin(x)/x;
  }
  else
  {
    static const double c_2=1.0/6.0;
    static const double c_4=1.0/120.0;
    static const double c_6=1.0/5040.0; 
    const double x_2 = x*x;
    const double x_4 = x_2*x_2;
    const double x_6 = x_2*x_2*x_2;
    return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
  }
}

const double jacobianTolerance = 1.0e-6;

TEST(googleTests, testMarginalization)
{
  // set the imu parameters
  ImuParams imuParameters;
  imuParameters.g_ = 9.81;
  imuParameters.sigma_g_ = 6.0e-4;
  imuParameters.sigma_a_ = 2.0e-3;
  imuParameters.sigma_b_g_ = 3.0e-6;
  imuParameters.sigma_b_a_ = 2.0e-5;
  imuParameters.b_a_tau_ = 3600.0;

  double imu_rate = 100.0;

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1,10.0);
  const double w_a_W_y = Eigen::internal::random(0.1,10.0);
  const double w_a_W_z = Eigen::internal::random(0.1,10.0);
  const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);

  const double duration = 0.5;
  const double dt = 1.0 / imu_rate;
  std::vector<ImuMeasurement> imuMeasurements0;
  std::vector<ImuMeasurement> imuMeasurements1;

  // states
  Eigen::Quaterniond q_ws;
  q_ws.setIdentity();
  Eigen::Vector3d v_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();

  // starting state
  Eigen::Quaterniond q_ws_0;
  Eigen::Quaterniond q_ws_0_true;
  Eigen::Vector3d v_s_0;
  Eigen::Vector3d v_s_0_true;
  Eigen::Vector3d b_g_0;
  Eigen::Vector3d b_a_0;
  double t0;

  // middle state
  Eigen::Quaterniond q_ws_1;
  Eigen::Vector3d v_s_1;
  Eigen::Vector3d b_g_1;
  Eigen::Vector3d b_a_1;
  double t1;

  // ending state
  Eigen::Quaterniond q_ws_2;
  Eigen::Vector3d v_s_2;
  Eigen::Vector3d b_g_2;
  Eigen::Vector3d b_a_2;
  double t2;

  
  // open file for gt state logging
  std::ofstream gt_file;
  // generate IMU measurements with noise
  for (int i = 0; i < int(duration*imu_rate); i++)
  {
    double time = double(i) / imu_rate;
    if (i == 10) // set as starting pose
    {
      q_ws_0 = q_ws;
      q_ws_0_true = q_ws;
      v_s_0 = v_s;
      v_s_0_true = v_s;
      b_g_0 = b_g;
      b_a_0 = b_a;
      t0 = time;
    }
    if (i == int((duration*imu_rate) / 2))
    {
      q_ws_1 = q_ws;
      v_s_1 = v_s;
      b_g_1 = b_g;
      b_a_1 = b_a;
      t1 = time;
    }
    if (i == int(duration*imu_rate) - 10) // set as end pose
    {
      q_ws_2 = q_ws;
      v_s_2 = v_s;
      b_g_2 = b_g;
      b_a_2 = b_a;
      t2 = time;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                    m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                    m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                    m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                    m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

    Eigen::Quaterniond dq;

    // propagate orientation
    const double theta_half = omega_S.norm()*dt*0.5;
    const double sinc_theta_half = sinc_test(theta_half); 
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half*0.5*dt*omega_S;
    dq.w() = cos_theta_half;
    q_ws = q_ws * dq;
    q_ws.normalize();

    // propagate speed
    v_w += dt*a_W; 
    v_s = q_ws.toRotationMatrix().inverse() * v_w;

    // log groundtruth states to file
    // generate measurements (with no noise)
    // TO DO: add biases
    ImuMeasurement new_meas;
    new_meas.t_ = time;
    new_meas.g_ = omega_S + imuParameters.sigma_g_/sqrt(dt)*Eigen::Vector3d::Random();
    new_meas.a_ = q_ws.toRotationMatrix().inverse() * (a_W - Eigen::Vector3d(0,0,imuParameters.g_)) + imuParameters.sigma_a_/sqrt(dt)*Eigen::Vector3d::Random();

    if (i < int((duration*imu_rate) / 2.0) + 5)
      imuMeasurements0.push_back(new_meas);

    if (i > int((duration*imu_rate) / 2.0) - 5)
      imuMeasurements1.push_back(new_meas);
  }

  Eigen::Quaterniond* q_ws_0_est = new Eigen::Quaterniond(q_ws_0);
  Eigen::Vector3d* v_s_0_est = new Eigen::Vector3d(v_s_0);
  Eigen::Vector3d* b_g_0_est = new Eigen::Vector3d(b_g_0);
  Eigen::Vector3d* b_a_0_est = new Eigen::Vector3d(b_a_0);

  Eigen::Quaterniond* q_ws_1_est = new Eigen::Quaterniond(q_ws_1);
  Eigen::Vector3d* v_s_1_est = new Eigen::Vector3d(v_s_1);
  Eigen::Vector3d* b_g_1_est = new Eigen::Vector3d(b_g_1);
  Eigen::Vector3d* b_a_1_est = new Eigen::Vector3d(b_a_1);

  Eigen::Quaterniond* q_ws_2_est = new Eigen::Quaterniond(q_ws_2);
  Eigen::Vector3d* v_s_2_est = new Eigen::Vector3d(v_s_2);
  Eigen::Vector3d* b_g_2_est = new Eigen::Vector3d(b_g_2);
  Eigen::Vector3d* b_a_2_est = new Eigen::Vector3d(b_a_2);
  
  // create ceres problem
  std::shared_ptr<ceres::Problem> problem 
    = std::shared_ptr<ceres::Problem>(new ceres::Problem());
  ceres::Solver::Options options;

  // add (fixed) initial state parameter blocks
  QuaternionParameterization* quat_param = new QuaternionParameterization;
  problem->AddParameterBlock(v_s_0_est->data(),3);
  problem->AddParameterBlock(q_ws_0_est->coeffs().data(),4);
  problem->AddParameterBlock(b_g_0_est->data(),3);
  problem->AddParameterBlock(b_a_0_est->data(),3);
  problem->SetParameterization(q_ws_0_est->coeffs().data(), quat_param);
  problem->AddParameterBlock(v_s_1_est->data(),3);
  problem->AddParameterBlock(q_ws_1_est->coeffs().data(),4);
  problem->AddParameterBlock(b_g_1_est->data(),3);
  problem->AddParameterBlock(b_a_1_est->data(),3);
  problem->SetParameterization(q_ws_1_est->coeffs().data(), quat_param);
  problem->AddParameterBlock(v_s_2_est->data(),3);
  problem->AddParameterBlock(q_ws_2_est->coeffs().data(),4);
  problem->AddParameterBlock(b_g_2_est->data(),3);
  problem->AddParameterBlock(b_a_2_est->data(),3);
  problem->SetParameterization(q_ws_2_est->coeffs().data(), quat_param);

  problem->SetParameterBlockConstant(b_a_0_est->data());
  problem->SetParameterBlockConstant(b_g_0_est->data());

  // create the IMU error terms
  
  ceres::CostFunction* imu_cost_func_0 = 
      new ImuVelocityCostFunction(t0, t1,
                                  imuMeasurements0,
                                  imuParameters);
  ceres::ResidualBlockId imu_id_0 = 
    problem->AddResidualBlock(imu_cost_func_0,
                            NULL,
                            q_ws_0_est->coeffs().data(),
                            v_s_0_est->data(),
                            b_g_0_est->data(),
                            b_a_0_est->data(),
                            q_ws_1_est->coeffs().data(),
                            v_s_1_est->data(),
                            b_g_1_est->data(),
                            b_a_1_est->data());

  ceres::CostFunction* imu_cost_func_1 = 
      new ImuVelocityCostFunction(t1, t2,
                                  imuMeasurements1,
                                  imuParameters);
  ceres::ResidualBlockId imu_id_1 =
    problem->AddResidualBlock(imu_cost_func_1,
                            NULL,
                            q_ws_1_est->coeffs().data(),
                            v_s_1_est->data(),
                            b_g_1_est->data(),
                            b_a_1_est->data(),
                            q_ws_2_est->coeffs().data(),
                            v_s_2_est->data(),
                            b_g_2_est->data(),
                            b_a_2_est->data());
  
  // create doppler measurements
  std::vector<std::pair<double,Eigen::Vector3d>> targets_0;
  std::vector<std::pair<double,Eigen::Vector3d>> targets_1;
  std::vector<std::pair<double,Eigen::Vector3d>> targets_2;

  double x = 10.0;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> d(0, 0.05);
  
  for (int y = -5; y < 5; y += 3)
  {
    for (int z = -5; z < 5; z += 3)
    {
      Eigen::Vector3d point;
      point << x, y, z;
      Eigen::Vector3d ray = point.normalized();

      Eigen::Vector3d v_target_0 = -1.0 * v_s_0;
      Eigen::Vector3d v_target_1 = -1.0 * v_s_1;
      Eigen::Vector3d v_target_2 = -1.0 * v_s_2;

      double doppler_0 = v_target_0.dot(ray) + d(gen);
      double doppler_1 = v_target_1.dot(ray) + d(gen);
      double doppler_2 = v_target_2.dot(ray) + d(gen);

      targets_0.push_back(std::pair<double,Eigen::Vector3d>(doppler_0, point));
      targets_1.push_back(std::pair<double,Eigen::Vector3d>(doppler_1, point));
      targets_2.push_back(std::pair<double,Eigen::Vector3d>(doppler_2, point));
    }
  }
  
  // create and add residuals and record their ids
  std::vector<ceres::ResidualBlockId> state_0_residuals;
  for (size_t i = 0; i < targets_0.size(); i++)
  {
    ceres::CostFunction* v_cost_func_0 = 
      new BodyVelocityCostFunction(targets_0[i].first,
                                   targets_0[i].second,
                                   0.01);
    ceres::ResidualBlockId id_0 = problem->AddResidualBlock(v_cost_func_0,
                                                         NULL,
                                                         v_s_0_est->data());
    state_0_residuals.push_back(id_0);
  }
  state_0_residuals.push_back(imu_id_0);

  std::vector<ceres::ResidualBlockId> state_1_residuals;
  for (size_t i = 0; i < targets_1.size(); i++)
  {
    ceres::CostFunction* v_cost_func_1 = 
      new BodyVelocityCostFunction(targets_1[i].first,
                                   targets_1[i].second,
                                   0.01);
    ceres::ResidualBlockId id_1 = problem->AddResidualBlock(v_cost_func_1,
                                                         NULL,
                                                         v_s_1_est->data());
    state_1_residuals.push_back(id_1);
  }
  state_1_residuals.push_back(imu_id_0);
  state_1_residuals.push_back(imu_id_1);

  std::vector<ceres::ResidualBlockId> state_2_residuals;
  for (size_t i = 0; i < targets_2.size(); i++)
  {
    ceres::CostFunction* v_cost_func_2 = 
      new BodyVelocityCostFunction(targets_2[i].first,
                                   targets_2[i].second,
                                   0.01);
    ceres::ResidualBlockId id_2 = problem->AddResidualBlock(v_cost_func_2,
                                                         NULL,
                                                         v_s_2_est->data());
    state_2_residuals.push_back(id_2);
  }
  state_2_residuals.push_back(imu_id_1);

  // solve the problem and save the state estimates
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem.get(), &summary);

  LOG(INFO) << summary.FullReport();

  Eigen::Quaterniond q_0_err = *q_ws_0_est * q_ws_0.inverse();
  Eigen::Vector3d v_0_err = *v_s_0_est - v_s_0;
  Eigen::Vector3d b_g_0_err = *b_g_0_est - b_g_0;
  Eigen::Vector3d b_a_0_err = *b_a_0_est - b_a_0;
  Eigen::Quaterniond q_1_err = *q_ws_1_est * q_ws_1.inverse();
  Eigen::Vector3d v_1_err = *v_s_1_est - v_s_1;
  Eigen::Vector3d b_g_1_err = *b_g_1_est - b_g_1;
  Eigen::Vector3d b_a_1_err = *b_a_1_est - b_a_1;
  Eigen::Quaterniond q_2_err = *q_ws_2_est * q_ws_2.inverse();
  Eigen::Vector3d v_2_err = *v_s_2_est - v_s_2;
  Eigen::Vector3d b_g_2_err = *b_g_2_est - b_g_2;
  Eigen::Vector3d b_a_2_err = *b_a_2_est - b_a_2;

  double err_lim = 1.0e-1;
  
  ASSERT_TRUE(q_0_err.coeffs().head(3).norm() < err_lim) << "orientation error at t0 of " 
                                        << q_0_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << q_ws_0_est->coeffs().transpose() << '\n'
                                        << "groundtruth: " << q_ws_0.coeffs().transpose();
  ASSERT_TRUE(v_0_err.norm() < err_lim) << "velocity error at t0 of " << v_0_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << v_s_0_est->transpose() << '\n'
                                        << "groundtruth: " << v_s_0.transpose();
  ASSERT_TRUE(b_g_0_err.norm() < err_lim) << "gyro bias error at t0 of " << b_g_0_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_g_0_est->transpose() << '\n'
                                        << "groundtruth: " << b_g_0.transpose();
  ASSERT_TRUE(b_a_0_err.norm() < err_lim) << "accel bias error at t0 of " << b_a_2_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_a_0_est->transpose() << '\n'
                                        << "groundtruth: " << b_a_0.transpose();

  ASSERT_TRUE(q_1_err.coeffs().head(3).norm() < err_lim) << "orientation error at t1 of " 
                                        << q_1_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << q_ws_1_est->coeffs().transpose() << '\n'
                                        << "groundtruth: " << q_ws_1.coeffs().transpose();
  ASSERT_TRUE(v_1_err.norm() < err_lim) << "velocity error at t1 of " << v_1_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << v_s_1_est->transpose() << '\n'
                                        << "groundtruth: " << v_s_1.transpose();
  ASSERT_TRUE(b_g_1_err.norm() < err_lim) << "gyro bias error at t1 of " << b_g_1_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_g_1_est->transpose() << '\n'
                                        << "groundtruth: " << b_g_1.transpose();
  ASSERT_TRUE(b_a_1_err.norm() < err_lim) << "accel bias error at t1 of " << b_a_1_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_a_1_est->transpose() << '\n'
                                        << "groundtruth: " << b_a_1.transpose();

  ASSERT_TRUE(q_2_err.coeffs().head(3).norm() < err_lim) << "orientation error at t2 of " 
                                        << q_2_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << q_ws_2_est->coeffs().transpose() << '\n'
                                        << "groundtruth: " << q_ws_2.coeffs().transpose();
  ASSERT_TRUE(v_2_err.norm() < err_lim) << "velocity error at t2 of " << v_2_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << v_s_2_est->transpose() << '\n'
                                        << "groundtruth: " << v_s_2.transpose();
  ASSERT_TRUE(b_g_2_err.norm() < err_lim) << "gyro bias error at t2 of " << b_g_2_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_g_2_est->transpose() << '\n'
                                        << "groundtruth: " << b_g_2.transpose();
  ASSERT_TRUE(b_a_2_err.norm() < err_lim) << "accel bias error at t2 of " << b_a_2_err.norm() 
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "  estimated: " << b_a_2_est->transpose() << '\n'
                                        << "groundtruth: " << b_a_2.transpose();
	
  /*---- marginalize out last state and associated measurements ----*/
  
  // save previous estimates
  Eigen::Quaterniond q_ws_1_prev = *q_ws_1_est;
  Eigen::Vector3d v_s_1_prev = *v_s_1_est;
  Eigen::Vector3d b_g_1_prev = *b_g_1_est;
  Eigen::Vector3d b_a_1_prev = *b_a_1_est;
  Eigen::Quaterniond q_ws_2_prev = *q_ws_2_est;
  Eigen::Vector3d v_s_2_prev = *v_s_2_est;
  Eigen::Vector3d b_g_2_prev = *b_g_2_est;
  Eigen::Vector3d b_a_2_prev = *b_a_2_est;

  // set up marginalization error term
  MarginalizationError* marginalization_error
      = new MarginalizationError(problem);

  // add residuals and states
  LOG(INFO) << "Adding " << state_0_residuals.size() << " residual blocks";
  marginalization_error->AddResidualBlocks(state_0_residuals);
  std::vector<double*> marginalize_param_blocks;
  marginalize_param_blocks.push_back(q_ws_0_est->coeffs().data());
  marginalize_param_blocks.push_back(v_s_0_est->data());
  marginalize_param_blocks.push_back(b_g_0_est->data());
  marginalize_param_blocks.push_back(b_a_0_est->data());
  LOG(INFO) << "Marginalizing parameter blocks"; // currently fails at this step
  marginalization_error->MarginalizeOut(marginalize_param_blocks);
  marginalization_error->UpdateErrorComputation();

  // add marginalization error to problem
  LOG(INFO) << "Adding marginalization residual to problem";
  problem->AddResidualBlock(marginalization_error, 
                           NULL, 
                           q_ws_1_est->coeffs().data(),
                           v_s_1_est->data(),
                           b_g_1_est->data(),
                           b_a_1_est->data());
  
  // check marginalization jacobians by finite differences
  double* parameters[4];
  parameters[0] = q_ws_1.coeffs().data();
  parameters[1] = v_s_1.data();
  parameters[2] = b_g_1.data();
  parameters[3] = b_a_1.data();

  double* jacobians_minimal[4];
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J0_min; // w.r.t. orientation at t1
  jacobians_minimal[0] = J0_min.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J1_min; // w.r.t. velocity at t1
  jacobians_minimal[1] = J1_min.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J2_min; // w.r.t. gyro bias at t1
  jacobians_minimal[2] = J2_min.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J3_min; // w.r.t. accel bias at t1
  jacobians_minimal[3] = J3_min.data();

  double* jacobians[4];
  Eigen::Matrix<double,12,4,Eigen::RowMajor> J0; // w.r.t. orientation at t1
  jacobians[0] = J0.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J1; // w.r.t. velocity at t1
  jacobians[1] = J1.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J2; // w.r.t. gyro bias at t1
  jacobians[2] = J2.data();
  Eigen::Matrix<double,12,3,Eigen::RowMajor> J3; // w.r.t. accel bias at t1
  jacobians[3] = J3.data();

  Eigen::Matrix<double,12,1> residuals;

  marginalization_error->EvaluateWithMinimalJacobians(parameters, 
                                                      residuals.data(), 
                                                      jacobians,
                                                      jacobians_minimal);

  double dx = 1e-6;

  Eigen::Matrix<double,12,4> J0_numDiff;
  Eigen::Matrix<double,12,3> J0_min_numDiff;
  for (size_t i=0; i<3; i++)
  {
    Eigen::Vector3d dp_0;
    Eigen::Matrix<double,12,1> residuals_p;
    Eigen::Matrix<double,12,1> residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    Eigen::Quaterniond q_ws_1_temp = q_ws_1;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    marginalization_error->Evaluate(parameters,residuals_p.data(),NULL);
    q_ws_1 = q_ws_1_temp; // reset to initial value
    dp_0[i] = -dx;
    quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
    marginalization_error->Evaluate(parameters,residuals_m.data(),NULL);
    q_ws_1 = q_ws_1_temp; // reset again
    J0_min_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
  }
  Eigen::Matrix<double,3,4,Eigen::RowMajor> J0_lift;
  quat_param->liftJacobian(parameters[0], J0_lift.data());

  J0_numDiff = J0_min_numDiff * J0_lift;
  if ((J0 - J0_numDiff).norm() > jacobianTolerance)
  {
    LOG(INFO) << "User provided Jacobian 0 does not agree with num diff:"
      << '\n' << "user provided J0: \n" << J0
      << '\n' << "\nnum diff J0: \n" << J0_numDiff  << "\n\n";
  }

  Eigen::Matrix<double,12,3> J1_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_1;
    Eigen::Matrix<double,12,1> residuals_p;
    Eigen::Matrix<double,12,1> residuals_m;
    dp_1.setZero();
    dp_1[i] = dx;
    Eigen::Vector3d v_s_1_temp = v_s_1; // save initial state
    v_s_1 = v_s_1 + dp_1;
    marginalization_error->Evaluate(parameters,residuals_p.data(),NULL);
    v_s_1 = v_s_1_temp; // reset
    v_s_1 = v_s_1 - dp_1;
    marginalization_error->Evaluate(parameters,residuals_m.data(),NULL);
    v_s_1 = v_s_1_temp;
    J1_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
  }
  
  if ((J1 - J1_numDiff).norm() > jacobianTolerance)
  {
    LOG(INFO) << "User provided jacobian 1 does not agree with num diff: "
      << "\nuser provided J1: \n" << J1 
      << "\n\nnum diff J1:\n" << J1_numDiff << "\n\n";
  }

  Eigen::Matrix<double,12,3> J2_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_2;
    Eigen::Matrix<double,12,1> residuals_p;
    Eigen::Matrix<double,12,1> residuals_m;
    dp_2.setZero();
    dp_2[i] = dx;
    Eigen::Vector3d b_g_1_temp = b_g_1; // save initial state
    b_g_1 = b_g_1 + dp_2;
    marginalization_error->Evaluate(parameters,residuals_p.data(),NULL);
    b_g_1 = b_g_1_temp; // reset
    b_g_1 = b_g_1 - dp_2;
    marginalization_error->Evaluate(parameters,residuals_m.data(),NULL);
    b_g_1 = b_g_1_temp;
    J2_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
  }
  
  if ((J2 - J2_numDiff).norm() > jacobianTolerance)
  {
    LOG(INFO) << "User provided jacobian 2 does not agree with num diff: "
      << "\nuser provided J2: \n" << J2 
      << "\n\nnum diff J2:\n" << J2_numDiff << "\n\n";
  }

  Eigen::Matrix<double,12,3> J3_numDiff;
  for (size_t i = 0; i < 3; i++)
  {
    Eigen::Vector3d dp_3;
    Eigen::Matrix<double,12,1> residuals_p;
    Eigen::Matrix<double,12,1> residuals_m;
    dp_3.setZero();
    dp_3[i] = dx;
    Eigen::Vector3d b_a_1_temp = b_a_1; // save initial state
    b_a_1 = b_a_1 + dp_3;
    marginalization_error->Evaluate(parameters,residuals_p.data(),NULL);
    b_a_1 = b_a_1_temp; // reset
    b_a_1 = b_a_1 - dp_3;
    marginalization_error->Evaluate(parameters,residuals_m.data(),NULL);
    b_a_1 = b_a_1_temp;
    J3_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
  }
  
  if ((J3 - J3_numDiff).norm() > jacobianTolerance)
  {
    LOG(INFO) << "User provided jacobian 3 does not agree with num diff: "
      << "\nuser provided J3: \n" << J3 
      << "\n\nnum diff J3:\n" << J3_numDiff << "\n\n";
  }
  
  // solve problem again and compare to earlier estimates
  ceres::Solve(options, problem.get(), &summary);

  LOG(INFO) << summary.FullReport();

  q_1_err = *q_ws_1_est * q_ws_2_prev.inverse();
  v_1_err = *v_s_1_est - v_s_1_prev;
  b_g_1_err = *b_g_1_est - b_g_1_prev;
  b_a_1_err = *b_a_1_est - b_a_1_prev;
  q_2_err = *q_ws_2_est * q_ws_2_prev.inverse();
  v_2_err = *v_s_2_est - v_s_2_prev;
  b_g_2_err = *b_g_2_est - b_g_2_prev;
  b_a_2_err = *b_a_2_est - b_a_2_prev;

  
  ASSERT_TRUE(q_1_err.coeffs().head(3).norm() < err_lim) 
                                        << "orientation error at t1 of " 
                                        << q_1_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << q_ws_1_est->coeffs().transpose() << '\n'
                                        << "full problem: " << q_ws_1_prev.coeffs().transpose();
  ASSERT_TRUE(v_1_err.norm() < err_lim) << "velocity error at t1 of " 
                                        << v_1_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << v_s_1_est->transpose() << '\n'
                                        << "full problem: " << v_s_1_prev.transpose();
  ASSERT_TRUE(b_g_1_err.norm() < err_lim) << "gyro bias error at t1 of " 
                                        << b_g_1_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << b_g_1_est->transpose() << '\n'
                                        << "full problem: " << b_g_1_prev.transpose();
  ASSERT_TRUE(b_a_1_err.norm() < err_lim) << "accel bias error at t1 of " 
                                        << b_a_1_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << b_a_1_est->transpose() << '\n'
                                        << "full problem: " << b_a_1_prev.transpose();

  ASSERT_TRUE(q_2_err.coeffs().head(3).norm() < err_lim) 
                                        << "orientation error at t2 of " 
                                        << q_2_err.coeffs().head(3).norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << q_ws_2_est->coeffs().transpose() << '\n'
                                        << "full problem: " << q_ws_2_prev.coeffs().transpose();
  ASSERT_TRUE(v_2_err.norm() < err_lim) << "velocity error at t2 of " 
                                        << v_2_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << v_s_2_est->transpose() << '\n'
                                        << "full problem: " << v_s_2_prev.transpose();
  ASSERT_TRUE(b_g_2_err.norm() < err_lim) << "gyro bias error at t2 of " 
                                        << b_g_2_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << b_g_2_est->transpose() << '\n'
                                        << "full problem: " << b_g_2_prev.transpose();
  ASSERT_TRUE(b_a_2_err.norm() < err_lim) << "accel bias error at t2 of " 
                                        << b_a_2_err.norm()
                                        << " is greater than the tolerance of " 
                                        << err_lim << "\n"
                                        << "marginalized: " << b_a_2_est->transpose() << '\n'
                                        << "full problem: " << b_a_2_prev.transpose();
  
  delete q_ws_1_est;
  delete v_s_1_est;
  delete b_a_1_est;
  delete b_g_1_est;
  
  delete q_ws_2_est;
  delete v_s_2_est;
  delete b_a_2_est;
  delete b_g_2_est;
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}