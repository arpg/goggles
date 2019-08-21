#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include "CeresCostFunctions.h"
#include "QuaternionParameterization.h"
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <fstream>

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

const double jacobianTolerance = 1.0e-3;

TEST(goggleTests, ImuIntegration)
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

  const double duration = 1.0;
  const double dt = 1.0 / imu_rate;
  std::vector<ImuMeasurement> imuMeasurements;

  // states
  Eigen::Quaterniond q_ws;
  q_ws.setIdentity();
  Eigen::Vector3d v_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();

  // starting state
  Eigen::Quaterniond q_ws_0;
  Eigen::Vector3d v_s_0;
  Eigen::Vector3d b_g_0;
  Eigen::Vector3d b_a_0;
  double t0;

  // ending state
  Eigen::Quaterniond q_ws_1;
  Eigen::Vector3d v_s_1;
  Eigen::Vector3d b_g_1;
  Eigen::Vector3d b_a_1;
  double t1;
	
	// open file for gt state logging
	std::ofstream gt_file;
  // generate IMU measurements with noise
  for (int i = 0; i < int(duration*imu_rate); i++)
  {
    double time = double(i) / imu_rate;
    if (i == 10) // set as starting pose
    {
      q_ws_0 = q_ws;
      v_s_0 = v_s;
      b_g_0 = b_g;
      b_a_0 = b_a;
      t0 = time;
    }
    if (i == int(duration*imu_rate) - 10) // set as end pose
    {
      q_ws_1 = q_ws;
      v_s_1 = v_s;
      b_g_1 = b_g;
      b_a_1 = b_a;
      t1 = time;
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

    imuMeasurements.push_back(new_meas);
  }
  q_ws = q_ws_0;
  v_s = v_s_0;
  b_g = b_g_0;
  b_a = b_a_0;
	std::cout << "q_1: " << q_ws_1.coeffs().transpose() << std::endl;
	// create ceres problem
	ceres::Problem problem;
	ceres::Solver::Options options;
	/*
	options.check_gradients = true;
	options.gradient_check_relative_precision = jacobianTolerance;
	*/
	// add (fixed) initial state parameter blocks
	ceres::LocalParameterization* quat_param = new QuaternionParameterization;
	problem.AddParameterBlock(v_s_0.data(),3);
	problem.AddParameterBlock(q_ws_0.coeffs().data(),4);
	problem.AddParameterBlock(b_g_0.data(),3);
	problem.AddParameterBlock(b_a_0.data(),3);
	//problem.SetParameterBlockConstant(v_s_0.data());
	//problem.SetParameterBlockConstant(q_ws_0.coeffs().data());
	problem.SetParameterization(q_ws_0.coeffs().data(), quat_param);
	//problem.SetParameterBlockConstant(b_g_0.data());
	//problem.SetParameterBlockConstant(b_a_0.data());
	// add variable parameter blocks for the final state
	problem.AddParameterBlock(v_s.data(),3);
	problem.AddParameterBlock(q_ws.coeffs().data(),4);
	problem.AddParameterBlock(b_g.data(),3);
	problem.AddParameterBlock(b_a.data(),3);
	problem.SetParameterization(q_ws.coeffs().data(), quat_param);

	// create the IMU error term
	ceres::CostFunction* imu_cost_func = 
			new ImuVelocityCostFunction(t0, t1,
																	imuMeasurements,
																	imuParameters);
	problem.AddResidualBlock(imu_cost_func,
													 NULL,
													 q_ws_0.coeffs().data(),
													 v_s_0.data(),
													 b_g_0.data(),
													 b_a_0.data(),
													 q_ws.coeffs().data(),
													 v_s.data(),
													 b_g.data(),
													 b_a.data());

	// create velocity measurement terms
	Eigen::Vector3d v0_meas = v_s_0 + 1.0e-2*Eigen::Vector3d::Random();
	Eigen::Vector3d v1_meas = v_s_1 + 1.0e-2*Eigen::Vector3d::Random();
	ceres::CostFunction* v0_cost = 
			new VelocityMeasCostFunction(v0_meas);
	problem.AddResidualBlock(v0_cost,
													 NULL,
													 v_s_0.data());
	ceres::CostFunction* v1_cost = 
			new VelocityMeasCostFunction(v1_meas);
	problem.AddResidualBlock(v1_cost,
													 NULL,
													 v_s_1.data());

	// check jacobians by manual inspection
	// automatic checking is not reliable because the information
	// matrix is a function of the states
	double* parameters[8];
	parameters[0] = q_ws_0.coeffs().data();
	parameters[1] = v_s_0.data();
	parameters[2] = b_g_0.data();
	parameters[3] = b_a_0.data();
	parameters[4] = q_ws.coeffs().data();
	parameters[5] = v_s.data();
	parameters[6] = b_g.data();
	parameters[7] = b_a.data();

	double* jacobians[8];
	Eigen::Matrix<double,12,4,Eigen::RowMajor> J0; // w.r.t. orientation at t0
	jacobians[0] = J0.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J1; // w.r.t. velocity at t0
	jacobians[1] = J1.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J2; // w.r.t. gyro bias at t0
	jacobians[2] = J2.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J3; // w.r.t. accel bias at t0
	jacobians[3] = J3.data();
	Eigen::Matrix<double,12,4,Eigen::RowMajor> J4; // w.r.t. orientation at t1
	jacobians[4] = J4.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J5; // w.r.t. velocity at t1
	jacobians[5] = J5.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J6; // w.r.t. gyro bias at t1
	jacobians[6] = J6.data();
	Eigen::Matrix<double,12,3,Eigen::RowMajor> J7; // w.r.t. accel bias at t1
	jacobians[7] = J7.data();

	Eigen::Matrix<double,12,1> residuals;

	imu_cost_func->Evaluate(parameters, residuals.data(), jacobians);

	double dx = 1e-6;

	Eigen::Matrix<double,12,3> J0_numDiff;
	for (size_t i=0; i<3; i++)
	{
		Eigen::Vector3d dp_0;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_0.setZero();
		dp_0[i] = dx;
		Eigen::Quaterniond q_ws_0_temp = q_ws_0;
		quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		q_ws_0 = q_ws_0_temp; // reset to initial value
		dp_0[i] = -dx;
		quat_param->Plus(parameters[0],dp_0.data(),parameters[0]);
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		q_ws_0 = q_ws_0_temp; // reset again
		J0_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	Eigen::Matrix<double,3,4,Eigen::RowMajor> J0_lift;
	QuaternionParameterization qp;
	qp.liftJacobian(parameters[0], J0_lift.data());
	if ((J0 - J0_numDiff * J0_lift).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided Jacobian 0 does not agree with num diff:"
			<< '\n' << "user provided J0: \n" << J0
			<< '\n' << "\nnum diff J0: \n" << J0_numDiff * J0_lift << '\n';
	}

	Eigen::Matrix<double,12,3> J1_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_1;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_1.setZero();
		dp_1[i] = dx;
		Eigen::Vector3d v_s_0_temp = v_s_0; // save initial state
		v_s_0 = v_s_0 + dp_1;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		v_s_0 = v_s_0_temp; // reset
		v_s_0 = v_s_0 - dp_1;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		v_s_0 = v_s_0_temp;
		J1_numDiff.col(i) = (residuals_p - residuals_m) / (2.0*dx);
	}
	if ((J1 - J1_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 1 does not agree with num diff: "
			<< "\nuser provided J1: \n" << J1 
			<< "\n\nnum diff J1:\n" << J1_numDiff << '\n';
	}

	Eigen::Matrix<double,12,3> J2_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_2;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_2.setZero();
		dp_2[i] = dx;
		Eigen::Vector3d b_g_0_temp = b_g_0;
		b_g_0 = b_g_0 + dp_2;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		b_g_0 = b_g_0_temp;
		b_g_0 = b_g_0 - dp_2;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		b_g_0 = b_g_0_temp;
		J2_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	if ((J2 - J2_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 2 does not agree with num diff:"
			<< "\nuser provided J2: \n" << J2
			<< "\n\nnum diff J2:\n" << J2_numDiff << '\n';
	}

	Eigen::Matrix<double,12,3> J3_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_3;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_3.setZero();
		dp_3[i] = dx;
		Eigen::Vector3d b_a_0_temp = b_a_0;
		b_a_0 = b_a_0 + dp_3;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		b_a_0 = b_a_0_temp;
		b_a_0 = b_a_0 - dp_3;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		b_a_0 = b_a_0_temp;
		J3_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	if ((J3 - J3_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 3 does not agree with num diff:"
			<< "\nuser provided J3:\n" << J3
			<< "\n\nnum diff J3:\n" << J3_numDiff << '\n';
	}

	Eigen::Matrix<double,12,3> J4_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_4;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_4.setZero();
		dp_4[i] = dx;
		Eigen::Quaterniond q_ws_temp = q_ws;
		quat_param->Plus(parameters[4],dp_4.data(),parameters[4]);
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		q_ws = q_ws_temp;
		dp_4[i] = -dx;
		quat_param->Plus(parameters[4], dp_4.data(), parameters[4]);
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		q_ws = q_ws_temp;
		J4_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	Eigen::Matrix<double,3,4> J4_lift;
	qp.liftJacobian(parameters[4], J4_lift.data());
	if ((J4 - (J4_numDiff * J4_lift)).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 4 does not agree with num diff:\n"
			<< "\nuser provided J4:\n" << J4
			<< "\n\nnum diff J5:\n" << J4_numDiff * J4_lift << '\n';
	}
	Eigen::Matrix<double,12,3> J5_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_5;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_5.setZero();
		dp_5[i] = dx;
		Eigen::Vector3d v_s_temp = v_s;
		v_s = v_s + dp_5;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		v_s = v_s_temp;
		v_s = v_s - dp_5;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		v_s = v_s_temp;
		J5_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	if ((J5 - J5_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 5 does not agree with num diff:\n"
			<< "\nuser provided J5:\n" << J5
			<< "\n\nnum diff J5:\n" << J5_numDiff << '\n';
	}
	Eigen::Matrix<double,12,3> J6_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_6;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_6.setZero();
		dp_6[i] = dx;
		Eigen::Vector3d b_g_temp = b_g;
		b_g = b_g + dp_6;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		b_g = b_g_temp;
		b_g = b_g - dp_6;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		b_g = b_g_temp;
		J6_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	if ((J6 - J6_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 6 does not agree with num diff:\n"
			<< "\nuser provided J6:\n" << J6
			<< "\n\nnum diff J6:\n" << J6_numDiff << '\n';
	}
	Eigen::Matrix<double,12,3> J7_numDiff;
	for (size_t i = 0; i < 3; i++)
	{
		Eigen::Vector3d dp_7;
		Eigen::Matrix<double,12,1> residuals_p;
		Eigen::Matrix<double,12,1> residuals_m;
		dp_7.setZero();
		dp_7[i] = dx;
		Eigen::Vector3d b_a_temp = b_a;
		b_a = b_a + dp_7;
		imu_cost_func->Evaluate(parameters,residuals_p.data(),NULL);
		b_a = b_a_temp;
		b_a = b_a - dp_7;
		imu_cost_func->Evaluate(parameters,residuals_m.data(),NULL);
		b_a = b_a_temp;
		J7_numDiff.col(i) = (residuals_p - residuals_m) / (2.0 * dx);
	}
	if ((J7 - J7_numDiff).norm() > jacobianTolerance)
	{
		LOG(ERROR) << "User provided jacobian 7 does not agree with num diff:\n"
			<< "\nuser provided J7:\n" << J7
			<< "\n\nnum diff J7:\n" << J7_numDiff << '\n';
	}
	// run the solver
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;	
	std::cout << "q_1_hat: " << q_ws.coeffs().transpose() << std::endl;
  // compare groundtruth states at t1 to states at t1 from imu integration
  double err_lim = 1.0e-1;
  
  Eigen::Vector3d v_err = v_s_1 - v_s;
  ASSERT_TRUE(v_err.norm() < err_lim) << "velocity error of " << v_err.norm() 
                                    << " is greater than the tolerance of " 
                                    << err_lim << "\n"
                                    << "  estimated: " << v_s.transpose() << '\n'
                                    << "groundtruth: " << v_s_1.transpose();

  
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
