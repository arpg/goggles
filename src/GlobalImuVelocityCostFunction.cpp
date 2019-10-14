#include <GlobalImuVelocityCostFunction.h>

/** 
  * @brief Constructor
  * @param[in] t0 Start time
  * @param[in] t1 End time
  * @param[in] imu_measurements Vector of imu measurements from before t0 to after t1
  * @param[in] imu_parameters Imu noise, bias, and rate parameters
  */
GlobalImuVelocityCostFunciton::GlobalImuVelocityCostFunciton(
  const double t0, const double t1,
  const std::vector<ImuMeasurement> &imu_measurements,
  const ImuParams &imu_parameters)
{
  set_num_residuals(12);
  mutable_parameter_block_sizes()->push_back(4); // orientation at t0 (i,j,k,w)
  mutable_parameter_block_sizes()->push_back(3); // velocity at t0
  mutable_parameter_block_sizes()->push_back(3); // imu gyro biases at t0
  mutable_parameter_block_sizes()->push_back(3); // imu accelerometer biases at t0
  mutable_parameter_block_sizes()->push_back(4); // orientation at t1
  mutable_parameter_block_sizes()->push_back(3); // velocity at t1
  mutable_parameter_block_sizes()->push_back(3); // imu gyro biases at t1
  mutable_parameter_block_sizes()->push_back(3); // imu accelerometer biases at t1

  SetImuMeasurements(imu_measurements);
  SetImuParameters(imu_parameters);
  SetT0(t0);
  SetT1(t1);

  if (t0 >= imu_measurements.front().t_)
    LOG(FATAL) << "First IMU measurement occurs after start time";
  if (t1 <= imu_measurements.back().t_)
    LOG(FATAL) << "Last IMU measurement occurs before end time";
}

~GlobalImuVelocityCostFunciton::GlobalImuVelocityCostFunciton();

/**
  * @brief Propagates orientation, velocity, and biases with given imu measurements
  * @remark This function can be used externally to propagate imu measurements
  * @param[in] imu_measurements Vector of imu measurements from before t0 to after t1
  * @param[in] imu_parameters Imu noise, bias, and rate parameters
  * @param[inout] orientation The starting orientation
  * @param[inout] velocity The starting velocity
  * @param[inout] gyro_bias The starting gyro bias
  * @param[inout] accel_bias The starting accelerometer bias
  * @param[in] t0 The starting time
  * @param[in] t1 The end time
  * @param[out] covariance Covariance for the given starting state
  * @param[out] jacobian Jacobian w.r.t. the starting state
  * @return The number of integration steps
  */
static int GlobalImuVelocityCostFunciton::Propagation(
  const std::vector<ImuMeasurement> &imu_measurements,
  const ImuParams &imu_parameters,
  Eigen::Quaterniond &orientation,
  Eigen::Vector3d &velocity,
  Eigen::Vector3d &gyro_bias,
  Eigen::Vector3d &accel_bias,
  double t0, double t1,
  covariance_t* covariance = NULL,
  jacobian_t* jacobian = NULL)
{
  double time = t0;
  double end = t1;

  if (imu_measurements.front().t_ >= time 
      || imu_measurements.back().t_ <= end)
  {
    LOG(ERROR) << "given imu measurements do not cover given timespan";
    return -1;
  }

  // initial conditions
  Eigen::Vector3d v_0 = velocity;
  Eigen::Quaterniond q_WS_0 = orientation;
  Eigen::Matrix3d C_WS_0 = orientation.toRotationMatrix();

  // increments
  Eigen::Quaterniond Delta_q(1,0,0,0);
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d C_double_integral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();

  // may not be required if not estimating position
  Eigen::Vector3d acc_double_integral = Eigen::Vector3d::Zero(); 

  // cross matrix accumulation
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();

  // may not be required if not estimating position
  Eigen::Matrix3d vp_db_g = Eigen::Matrix3d::Zero();

  // Covariance of the increment without biases
  covariance_t P_delta = covariance_t::Zero();

  double delta_t = 0;
  int i = 0;
  for (std::vector<ImuMeasurement>::const_iterator it = imu_measurements.begin();
    it != imu_measurements.end(); it++)
  {
    Eigen::Vector3d omega_S_0 = it->g_;
    Eigen::Vector3d acc_S_0 = it->a_;
    time = it->t_;
    Eigen::Vector3d omega_S_1 = (it+1)->g_;
    Eigen::Vector3d acc_S_1 = (it+1)->a_;

    double next_time;
    if ((it+1) == imu_measurements.end())
      next_time = t1;
    else
      next_time = (it+1)->t_;

    double dt = next_time - time;

    // if both measurements are before t0 or both measurements are
    // after t1, skip ahead
    if (next_time < t0 || time > t1)
      continue;
    // if first measurement is out of time range but second is within,
    // interpolate first measurement at t0
    if (t0 > time)
    {
      double interval = next_time - time;
      time = t0;
      dt = next_time - time;
      const double r = 1.0 - dt / interval;
      omega_S_0 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_0 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }
    // if second measurement is out of time range but first is within,
    // interpolate second measurement at t1
    if (end < next_time)
    {
      double interval = next_time - it->t_;
      next_time = t1;
      dt = next_time - time;
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    delta_t += dt;

    double sigma_g_c = imu_parameters.sigma_g_;
    dobule sigma_a_c = imu_parameters.sigma_a_;

    // check for gyro and accelerometer saturation
    bool gyro_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(omega_S_0[i]) > imu_parameters.g_max_
        || fabs(omega_S_1[i]) > imu_parameters.g_max_)
        gyro_saturation = true;
    }
    if (gyro_saturation)
      LOG(WARNING) "gyro saturation";

    bool accel_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(acc_S_0[i]) > imu_parameters.a_max_
        || fabs(acc_S_1[i]) > imu_parameters.a_max_)
        accel_saturation = true;
    }
    if (accel_saturation)
      LOG(WARNING) "accelerometer saturation";

    // actual propagation:
    QuaternionParameterization qp;
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omege_S_true = (0.5*(omega_S_0+omega_S_1) - gyro_bias);
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = qp.sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;

    // rotation matrix integrals
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral + 0.5 * (C + C_1) * dt;

    // rotation matrix double integrals
    // may not be needed if not estimating position
    C_double_integral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
    acc_double_integral += acc_integral * dt + 0.25*(C+C_1)*acc_S_true*dt*dt;

    // Jacobian parts
    dalpha_db_g += dt * C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross
      + RightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = CrossMatrix(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g + 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

    // this component may not be required if we're not estimating position
    dp_db_g += dt * dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

    // covariance propagation
    if (covariance)
    {
      jacobian_t F_delta = jacobian_t::Identity();

      F_delta.block<3,3>(0,6) = -dt * C_1;
      F_delta.block<3,3>(3,0) = CrossMatrix(0.5*(C+C_1)*acc_S_true*dt);
      F_delta.block<3,3>(3,6) = 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
      F_delta.block<3,3>(3,9) = -0.5*(C+C_1)*dt;

      P_delta = F_delta * P_delta * F_delta.transpose();

      // add noise
      // assuming isotropic noise, so no transform is necessary
      const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
      P_delta(0,0) += sigma2_dalpha;
      P_delta(1,1) += sigma2_dalpha;
      P_delta(2,3) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * sigma_a_c;
      P_delta(3,3) += sigma2_v;
      P_delta(4,4) += sigma2_v;
      P_delta(5,5) += sigma2_v;
      const double sigma2_b_g = dt * imu_parameters.sigma_b_g_ * imu_parameters.sigma_b_g_;
      P_delta(6,6) += sigma2_b_g;
      P_delta(7,7) += sigma2_b_g;
      P_delta(8,8) += sigma2_b_g;
      const double sigma2_b_a = dt * imu_parameters.sigma_b_a_ * imu_parameters.sigma_b_a_;
      P_delta(9,9) += sigma2_b_a;
      P_delta(10,10) += sigma2_b_a;
      P_delta(11,11) += sigma2_b_a;
    }

    // update persistent components
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;
    cross = cross_1;
    dv_db_g = dv_db_g_1;
    time = next_time;

    i++;

    if (next_time == t1)
      break;
  }

  // propagation output
  const Eigen::Vector3d g_W = imu_params.g * Eigen::Vector3d(0,0,1);
  orientation = q_WS_0 * Delta_q;
  velocity += C_WS_0 * acc_integral - g_W * Delta_t;

  // assign jacobian if requested
  if (jacobian)
  {
    jacobian_t &F = *jacobian;
    F.setIdentity();
    F.block<3,3>(0,6) = -C_WS_0 * dalpha_db_g;
    F.block<3,3>(3,0) = -CrossMatrix(C_WS_0 * acc_integral);
    F.block<3,3>(3,6) = C_WS_0 * dv_db_g;
    F.block<3,3>(3,9) = -C_WS_0 * C_integral;
  }

  // assign overall covariance if requested
  if (covariance)
  {
    covariance_t &P = *covariance;
    covariance_t T = covariance_t::Identity();
    T.topLeftCorner<3,3>() = C_WS_0;
    T.block<3,3>(3,3) = C_WS_0;
    P = T * P_delta * T.transpose();
  }

  return i;
}

/**
  * @brief Propagates orientation, velocity, and biases with given imu measurements
  * @param[in] orientation The starting orientation
  * @param[in] gyro_bias The starting gyro bias
  * @param[in] accel_bias The starting accelerometer bias
  */
int GlobalImuVelocityCostFunciton::RedoPreintegration(
  const Eigen::Quaterniond &orientation,
  const Eigen::Vector3d &velocity,
  const Eigen::Vector3d &gyro_bias,
  const Eigen::Vector3d &accel_bias);

/**
  * @brief Evaluate the error term and compute jacobians
  * @param parameters Pointer to the parameters
  * @param residuals Pointer to the residuals
  * @param jacobians Pointer to the jacobians
  * @return Success of the evaluation
  */
bool GlobalImuVelocityCostFunciton::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const;

/**
  * @brief Evaluate the error term and compute jacobians in both full and minimal form
  * @param parameters Pointer to the parameters
  * @param residuals Pointer to the residuals
  * @param jacobians Pointer to the jacobians
  * @param jacobians_minimal Pointer to the minimal jacobians
  * @return Success of the evaluation
  */
bool GlobalImuVelocityCostFunciton::EvaluateWithMinimalJacobians(
  double const* const* parameters, 
  double* residuals, 
  double** jacobians,
  double** jacobians_minimal) const;

size_t GlobalImuVelocityCostFunciton::ResidualDim() const
{
  return num_residuals();
}

Eigen::Matrix3d GlobalImuVelocityCostFunciton::CrossMatrix(
  Eigen::Vector3d &in_vec)
{
  Eigen::Matrix3d cross_mx;
  cross_mx <<         0, -in_vec(2),  in_vec(1),
              in_vec(2),          0, -in_vec(0),
             -in_vec(1),  in_vec(0),          0;

  return cross_mx;
}

Eigen::Matrix3d GlobalImuVelocityCostFunciton::RightJacobian(
  Eigen::Vector3d &in_vec)
{

}