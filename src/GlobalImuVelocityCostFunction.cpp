#include <GlobalImuVelocityCostFunction.h>

inline Eigen::Matrix3d CrossMatrix(const Eigen::Vector3d &in_vec)
{
  Eigen::Matrix3d cross_mx;
  cross_mx <<         0, -in_vec(2),  in_vec(1),
              in_vec(2),          0, -in_vec(0),
             -in_vec(1),  in_vec(0),          0;
  return cross_mx;
}

inline Eigen::Matrix3d RightJacobian(const Eigen::Vector3d &phi_vec)
{
  const double Phi = phi_vec.norm();
  Eigen::Matrix3d right_jacobian = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d Phi_x = CrossMatrix(phi_vec);
  const Eigen::Matrix3d Phi_x2 = Phi_x * Phi_x;
  if (Phi < 1.0e-4)
  {
    right_jacobian += -0.5*Phi_x + 1.0/6.0*Phi_x2;
  }
  else
  {
    const double Phi2 = Phi*Phi;
    const double Phi3 = Phi*Phi2;
    right_jacobian += (-(1.0-cos(Phi))/Phi2)*Phi_x + ((Phi-sin(Phi))/Phi3)*Phi_x2;
  }
  return right_jacobian;
}

/** 
  * @brief Constructor
  * @param[in] t0 Start time
  * @param[in] t1 End time
  * @param[in] imu_measurements Vector of imu measurements from before t0 to after t1
  * @param[in] imu_parameters Imu noise, bias, and rate parameters
  */
GlobalImuVelocityCostFunction::GlobalImuVelocityCostFunction(
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

  if (t0 < imu_measurements.front().t_)
    LOG(FATAL) << "First IMU measurement occurs after start time";
  if (t1 > imu_measurements.back().t_)
    LOG(FATAL) << "Last IMU measurement occurs before end time";
}

GlobalImuVelocityCostFunction::~GlobalImuVelocityCostFunction(){}

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
int GlobalImuVelocityCostFunction::Propagation(
  const std::vector<ImuMeasurement> &imu_measurements,
  const ImuParams &imu_parameters,
  Eigen::Quaterniond &orientation,
  Eigen::Vector3d &velocity,
  Eigen::Vector3d &gyro_bias,
  Eigen::Vector3d &accel_bias,
  double t0, double t1,
  covariance_t* covariance,
  jacobian_t* jacobian)
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
  //Eigen::Matrix3d C_double_integral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();

  // may not be required if not estimating position
  //Eigen::Vector3d acc_double_integral = Eigen::Vector3d::Zero(); 

  // cross matrix accumulation
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();

  // may not be required if not estimating position
  //Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

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
    if (next_time <= t0 || time >= t1)
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
    double sigma_a_c = imu_parameters.sigma_a_;

    // check for gyro and accelerometer saturation
    bool gyro_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(omega_S_0[i]) > imu_parameters.g_max_
        || fabs(omega_S_1[i]) > imu_parameters.g_max_)
        gyro_saturation = true;
    }
    if (gyro_saturation)
      LOG(WARNING) << "gyro saturation";

    bool accel_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(acc_S_0[i]) > imu_parameters.a_max_
        || fabs(acc_S_1[i]) > imu_parameters.a_max_)
        accel_saturation = true;
    }
    if (accel_saturation)
      LOG(WARNING) << "accelerometer saturation";

    // actual propagation:
    QuaternionParameterization qp;
    Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1)-gyro_bias);
    Eigen::Quaterniond dq = qp.DeltaQ(omega_S_true * dt);
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;

    // rotation matrix integrals
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5*(acc_S_0+acc_S_1)-accel_bias);
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral + 0.5 * (C + C_1)*acc_S_true*dt;

    // rotation matrix double integrals
    // may not be needed if not estimating position
    //C_double_integral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
    //acc_double_integral += acc_integral * dt + 0.25*(C+C_1)*acc_S_true*dt*dt;

    // Jacobian parts
    dalpha_db_g += dt * C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross
      + RightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = CrossMatrix(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g + 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

    // this component may not be required if we're not estimating position
    //dp_db_g += dt * dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

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
  const Eigen::Vector3d g_W = imu_parameters.g_ * Eigen::Vector3d(0,0,1);
  orientation = q_WS_0 * Delta_q;
  velocity += C_WS_0 * acc_integral - g_W * delta_t;

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
int GlobalImuVelocityCostFunction::RedoPreintegration(
  const Eigen::Quaterniond &orientation,
  const Eigen::Vector3d &velocity,
  const Eigen::Vector3d &gyro_bias,
  const Eigen::Vector3d &accel_bias) const
{
  std::lock_guard<std::mutex> lock(preintegration_mutex_);

  double time = t0_;
  double end = t1_;

  if (imu_measurements_.front().t_ >= time 
      || imu_measurements_.back().t_ <= end)
  {
    LOG(ERROR) << "given imu measurements do not cover given timespan";
    return -1;
  }

  // initialize increments
  Delta_q_ = Eigen::Quaterniond(1,0,0,0);
  C_integral_ = Eigen::Matrix3d::Zero();
  //C_double_integral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  //acc_double_integral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulation
  cross_ = Eigen::Matrix3d::Zero();

  // sub jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();

  // Covariance of the increment
  P_delta_ = covariance_t::Zero();

  double delta_t = 0;
  int i = 0;
  for (std::vector<ImuMeasurement>::const_iterator it = imu_measurements_.begin();
    it != imu_measurements_.end(); it++)
  {
    Eigen::Vector3d omega_S_0 = it->g_;
    Eigen::Vector3d acc_S_0 = it->a_;
    time = it->t_;
    Eigen::Vector3d omega_S_1 = (it+1)->g_;
    Eigen::Vector3d acc_S_1 = (it+1)->a_;

    double next_time;
    if ((it+1) == imu_measurements_.end())
      next_time = t1_;
    else
      next_time = (it+1)->t_;

    double dt = next_time - time;

    // if both measurements are before t0 or both measurements are
    // after t1, skip ahead
    if (next_time <= t0_ || time >= t1_)
      continue;
    // if first measurement is out of time range but second is within,
    // interpolate first measurement at t0
    if (t0_ > time)
    {
      double interval = next_time - time;
      time = t0_;
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
      next_time = t1_;
      dt = next_time - time;
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    delta_t += dt;

    double sigma_g_c = imu_parameters_.sigma_g_;
    double sigma_a_c = imu_parameters_.sigma_a_;

    // check for gyro and accelerometer saturation
    bool gyro_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(omega_S_0[i]) > imu_parameters_.g_max_
        || fabs(omega_S_1[i]) > imu_parameters_.g_max_)
        gyro_saturation = true;
    }
    if (gyro_saturation)
      LOG(WARNING) << "gyro saturation";

    bool accel_saturation = false;
    for (int i = 0; i < 3; i++)
    {
      if (fabs(acc_S_0[i]) > imu_parameters_.a_max_
        || fabs(acc_S_1[i]) > imu_parameters_.a_max_)
        accel_saturation = true;
    }
    if (accel_saturation)
      LOG(WARNING) << "accelerometer saturation";

    //LOG(ERROR) << "Delta_q_:\n" << Delta_q_.coeffs().transpose();

    // orientation propagation
    QuaternionParameterization qp;
    const Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1)-gyro_bias);
    Eigen::Quaterniond dq = qp.DeltaQ(omega_S_true * dt);
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;

    // rotation matrix and acceleration integral
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1) - accel_bias);
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5*(C+C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
      + 0.5 * (C + C_1) * acc_S_true * dt;

    // double integrals
    //C_double_integral_ += C_integral_* dt + 0.25 * (C + C_1) * dt * dt;
    //acc_double_integral_ += acc_integral_ * dt 
    // + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // jacobian parts
    dalpha_db_g_ += C_1 * RightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
      + RightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = CrossMatrix(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
      + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    //dp_db_g += dt * dv_db_g_
    //  + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    // covariance propagation
    jacobian_t F_delta = jacobian_t::Identity();
    F_delta.block<3,3>(0,6) = -dt * C_1;
    F_delta.block<3,3>(3,0) = CrossMatrix(0.5 * (C + C_1) * acc_S_true * dt);
    F_delta.block<3,3>(3,6) = 0.5 * dt 
      * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3,3>(3,9) = -0.5 * (C + C_1) * dt;
    P_delta_ = F_delta * P_delta_ * F_delta.transpose();

    // add noise 
    // no transforms since we're assuming isotropic noise
    const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
    P_delta_(0,0) += sigma2_dalpha;
    P_delta_(1,1) += sigma2_dalpha;
    P_delta_(2,2) += sigma2_dalpha;
    const double sigma2_v = dt * sigma_a_c * sigma_a_c;
    P_delta_(3,3) += sigma2_v;
    P_delta_(4,4) += sigma2_v;
    P_delta_(5,5) += sigma2_v;
    const double sigma2_b_g = dt * imu_parameters_.sigma_b_g_ * imu_parameters_.sigma_b_g_;
    P_delta_(6,6) += sigma2_b_g;
    P_delta_(7,7) += sigma2_b_g;
    P_delta_(8,8) += sigma2_b_g;
    const double sigma2_b_a = dt * imu_parameters_.sigma_b_a_ * imu_parameters_.sigma_b_a_;
    P_delta_(9,9) += sigma2_b_a;
    P_delta_(10,10) += sigma2_b_a;
    P_delta_(11,11) += sigma2_b_a;

    // update persistent quantities
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = next_time;

    i++;

    if (next_time == t1_)
      break;
  }

  // store the linearization point
  velocity_ref_ = velocity;

  // enforce symmetry
  P_delta_ = 0.5 * P_delta_ + 0.5 * P_delta_.transpose().eval();

  // calculate inverse
  information_ = P_delta_.inverse();
  information_ = 0.5 * information_ + 0.5 * information_.transpose().eval();

  // square root information
  Eigen::LLT<information_t> llt_of_information(information_);
  sqrt_information_ = llt_of_information.matrixL().transpose();

  return i;
}

/**
  * @brief Evaluate the error term and compute jacobians
  * @param parameters Pointer to the parameters
  * @param residuals Pointer to the residuals
  * @param jacobians Pointer to the jacobians
  * @return Success of the evaluation
  */
bool GlobalImuVelocityCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

/**
  * @brief Evaluate the error term and compute jacobians in both full and minimal form
  * @param parameters Pointer to the parameters
  * @param residuals Pointer to the residuals
  * @param jacobians Pointer to the jacobians
  * @param jacobians_minimal Pointer to the minimal jacobians
  * @return Success of the evaluation
  */
bool GlobalImuVelocityCostFunction::EvaluateWithMinimalJacobians(
  double const* const* parameters, 
  double* residuals, 
  double** jacobians,
  double** jacobians_minimal) const
{
  // get parameters
  const Eigen::Quaterniond q_WS_0(parameters[0][3], 
                                  parameters[0][0], 
                                  parameters[0][1],
                                  parameters[0][2]);
  
  const Eigen::Quaterniond q_WS_1(parameters[4][3],
                                  parameters[4][0],
                                  parameters[4][1],
                                  parameters[4][2]);

  Eigen::Vector3d v_W_0;
  Eigen::Vector3d v_W_1;
  Eigen::Vector3d gyro_bias_0;
  Eigen::Vector3d gyro_bias_1;
  Eigen::Vector3d accel_bias_0;
  Eigen::Vector3d accel_bias_1; 
  for (size_t i = 0; i < 3; i++)
  {
    v_W_0[i] = parameters[1][i];
    v_W_1[i] = parameters[5][i];
    gyro_bias_0[i] = parameters[2][i];
    gyro_bias_1[i] = parameters[6][i];
    accel_bias_0[i] = parameters[3][i];
    accel_bias_1[i] = parameters[7][i];
  }

  const Eigen::Matrix3d C_WS_0 = q_WS_0.toRotationMatrix();
  const Eigen::Matrix3d C_SW_0 = q_WS_0.inverse().toRotationMatrix();

  // redo preintegration if required
  const double Delta_t = t1_ - t0_;
  Eigen::Vector3d delta_b_g;
  {
    std::lock_guard<std::mutex> lock(preintegration_mutex_);
    delta_b_g = gyro_bias_0 - gyro_bias_ref_;
  }
  Eigen::Matrix<double,6,1> delta_b;
  delta_b.head(3) = delta_b_g;
  delta_b.tail(3) = accel_bias_0 - accel_bias_ref_;
  redo_ = redo_ || (delta_b_g.norm() * Delta_t > 1.0e-4);
  if (redo_)
  {
    RedoPreintegration(q_WS_0, v_W_0, gyro_bias_0, accel_bias_0);
    redo_counter_++;
    delta_b_g.setZero();
    redo_ = false;
  }

  // do propagation
  {
    std::lock_guard<std::mutex> lock(preintegration_mutex_);
    const Eigen::Vector3d g_W = imu_parameters_.g_ * Eigen::Vector3d(0,0,1);

    // assign jacobian w.r.t. x0
    jacobian_t F0 = jacobian_t::Identity();
    const Eigen::Vector3d delta_v_est_W = 
      v_W_0 - v_W_1 - g_W * Delta_t;
    QuaternionParameterization qp;
    const Eigen::Quaterniond Dq = qp.DeltaQ(-dalpha_db_g_ * delta_b_g)*Delta_q_;
    F0.block<3,3>(0,0) = (qp.oplus(Dq*q_WS_1.inverse())
      * qp.qplus(q_WS_0)).topLeftCorner(3,3);
    F0.block<3,3>(0,6) = (qp.qplus(q_WS_1.inverse() * q_WS_0)
      * qp.qplus(Dq)).topLeftCorner(3,3)*(-dalpha_db_g_);

    F0.block<3,3>(3,0) = C_SW_0 * CrossMatrix(delta_v_est_W);
    F0.block<3,3>(3,3) = C_SW_0;
    F0.block<3,3>(3,6) = dv_db_g_;
    F0.block<3,3>(3,9) = -C_integral_;

    // assign jacobian w.r.t. x1
    jacobian_t F1 = jacobian_t::Identity();
    F1.block<3,3>(0,0) = -(qp.oplus(Dq) 
      * qp.qplus(q_WS_0) 
      * qp.oplus(q_WS_1.inverse())).topLeftCorner(3,3);
    F1.block<3,3>(3,3) = -C_SW_0;

    // assign the error vector
    Eigen::Matrix<double,12,1> error;
    error.head(3) = 2.0*(Dq*(q_WS_1.inverse()*q_WS_0)).vec();
    error.segment<3>(3) = C_SW_0 * delta_v_est_W + acc_integral_ 
      + F0.block<3,6>(3,6) * delta_b;
    error.segment<3>(6) = gyro_bias_0 - gyro_bias_1;
    error.tail(3) = accel_bias_0 - accel_bias_1;

    // weight the error vector
    Eigen::Map<Eigen::Matrix<double,12,1>> weighted_error(residuals);
    weighted_error = sqrt_information_ * error;

    // assign the jacobians if requested
    if (jacobians != NULL)
    {
      // jacobian w.r.t. orientation at t0
      if (jacobians[0] != NULL)
      {
        Eigen::Matrix<double,12,3> J0_minimal = sqrt_information_ 
          * F0.topLeftCorner(12,3);

        // get lift jacobian 
        Eigen::Matrix<double,3,4> J_lift;
        qp.ComputeLiftJacobian(parameters[0], J_lift.data());

        // get overparameterized jacobian
        Eigen::Map<Eigen::Matrix<double,12,4,Eigen::RowMajor>> J0(jacobians[0]);
        J0 = J0_minimal * J_lift;

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[0] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J0_minimal_mapped(
              jacobians_minimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      // jacobian w.r.t. velocity at t0
      if (jacobians[1] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J1(jacobians[1]);
        J1 = sqrt_information_ * F0.block<12,3>(0,3);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[1] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J1_minimal_mapped(
              jacobians_minimal[1]);
            J1_minimal_mapped = J1;
          }
        }
      }
      // jacobian w.r.t. gyro bias at t0
      if (jacobians[2] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J2(jacobians[2]);
        J2 = sqrt_information_ * F0.block<12,3>(0,6);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[2] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J2_minimal_mapped(
              jacobians_minimal[2]);
            J2_minimal_mapped = J2;
          }
        }
      }
      // jacobian w.r.t. accel bias at t0
      if (jacobians[3] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J3(jacobians[3]);
        J3 = sqrt_information_ * F0.block<12,3>(0,9);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[3] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J3_minimal_mapped(
              jacobians_minimal[3]);
            J3_minimal_mapped = J3;
          }
        }
      }
      // jacobian w.r.t. orientation at t1
      if (jacobians[4] != NULL)
      {
        Eigen::Matrix<double,12,3> J4_minimal = sqrt_information_ 
          * F1.topLeftCorner(12,3);

        // get lift jacobian 
        Eigen::Matrix<double,3,4> J_lift;
        qp.ComputeLiftJacobian(parameters[4], J_lift.data());

        // get overparameterized jacobian
        Eigen::Map<Eigen::Matrix<double,12,4,Eigen::RowMajor>> J4(jacobians[4]);
        J4 = J4_minimal * J_lift;

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[4] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J4_minimal_mapped(
              jacobians_minimal[4]);
            J4_minimal_mapped = J4_minimal;
          }
        }
      }
      // jacobian w.r.t. velocity at t1
      if (jacobians[5] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J5(jacobians[5]);
        J5 = sqrt_information_ * F1.block<12,3>(0,3);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[5] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J5_minimal_mapped(
              jacobians_minimal[5]);
            J5_minimal_mapped = J5;
          }
        }
      }
      // jacobian w.r.t. gyro bias at t1
      if (jacobians[6] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J6(jacobians[6]);
        J6 = sqrt_information_ * F1.block<12,3>(0,6);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[6] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J6_minimal_mapped(
              jacobians_minimal[6]);
            J6_minimal_mapped = J6;
          }
        }
      }
      // jacobian w.r.t. accel bias at t1
      if (jacobians[7] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J7(jacobians[7]);
        J7 = sqrt_information_ * F1.block<12,3>(0,9);

        if (jacobians_minimal != NULL)
        {
          if (jacobians_minimal[7] != NULL)
          {
            Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J7_minimal_mapped(
              jacobians_minimal[7]);
            J7_minimal_mapped = J7;
          }
        }
      }
    }
  }
  return true;
}

size_t GlobalImuVelocityCostFunction::ResidualDim() const
{
  return num_residuals();
}

