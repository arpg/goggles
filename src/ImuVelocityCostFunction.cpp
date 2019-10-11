#include <ImuVelocityCostFunction.h>

ImuVelocityCostFunction::ImuVelocityCostFunction(double t0, 
  double t1, 
  std::vector<ImuMeasurement> &imu_measurements,
  ImuParams &params)
: t0_(t0), t1_(t1), imu_measurements_(imu_measurements), params_(params)
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
}

ImuVelocityCostFunction::~ImuVelocityCostFunction(){}

bool ImuVelocityCostFunction::Evaluate(double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool ImuVelocityCostFunction::EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const
{
      // map parameter blocks to eigen containers
  Eigen::Map<const Eigen::Quaterniond> q_ws_0(&parameters[0][0]);
  Eigen::Map<const Eigen::Vector3d> v_s_0(&parameters[1][0]);
  Eigen::Map<const Eigen::Vector3d> b_g_0(&parameters[2][0]);
  Eigen::Map<const Eigen::Vector3d> b_a_0(&parameters[3][0]);
  Eigen::Map<const Eigen::Quaterniond> q_ws_1(&parameters[4][0]);
  Eigen::Map<const Eigen::Vector3d> v_s_1(&parameters[5][0]);
  Eigen::Map<const Eigen::Vector3d> b_g_1(&parameters[6][0]);
  Eigen::Map<const Eigen::Vector3d> b_a_1(&parameters[7][0]);

      // get rotation matrices
  const Eigen::Matrix3d C_ws_0 = q_ws_0.toRotationMatrix();
  const Eigen::Matrix3d C_sw_0 = C_ws_0.inverse();
  const Eigen::Matrix3d C_ws_1 = q_ws_1.toRotationMatrix();
  const Eigen::Matrix3d C_sw_1 = C_ws_1.inverse();

      // initialize propagated states
  Eigen::Quaterniond q_ws_hat(q_ws_0);

  Eigen::Vector3d v_s_hat(v_s_0(0), v_s_0(1), v_s_0(2));
  Eigen::Vector3d b_a_hat(b_a_0(0), b_a_0(1), b_a_0(2));
  Eigen::Vector3d b_g_hat(b_g_0(0), b_g_0(1), b_g_0(2));
  Eigen::Matrix<double,12,12> F; // jacobian matrix
  Eigen::Matrix<double,12,12> P; // covariance matrix
  Eigen::Matrix<double,12,12> Q; // measurement noise matrix

  F.setIdentity();
  P.setIdentity();
  Q.setIdentity();

  // set up noise matrix
  Q.block<3,3>(0,0) *= params_.sigma_g_ * params_.sigma_g_;
  Q.block<3,3>(3,3) *= params_.sigma_a_ * params_.sigma_a_;
  Q.block<3,3>(6,6) *= params_.sigma_b_g_ * params_.sigma_b_g_;
  Q.block<3,3>(9,9) *= params_.sigma_b_a_ * params_.sigma_b_a_;
      
  // propagate imu measurements, tracking jacobians and covariance
  int num_meas = 0;
  for (int i = 1; i < imu_measurements_.size(); i++)
  {
    if (imu_measurements_[i].t_ > t0_ 
        && imu_measurements_[i-1].t_ < t1_)
    {
      num_meas++;

      ImuMeasurement meas_0 = imu_measurements_[i-1];
      ImuMeasurement meas_1 = imu_measurements_[i];

      // if meas_0 is before t0_, interpolate measurement to match t0_
      if (meas_0.t_ < t0_)
      {
        double c = (t0_ - meas_0.t_) / (meas_1.t_ - meas_0.t_);
        meas_0.t_ = t0_;
        meas_0.g_ = ((1.0 - c) * meas_0.g_ + c * meas_1.g_).eval();
        meas_0.a_ = ((1.0 - c) * meas_0.a_ + c * meas_1.a_).eval();
      }

      // if meas_1 is after t1_, interpolate measurement to match t1_
      if (meas_1.t_ > t1_)
      {
        double c = (t1_ - meas_0.t_) / (meas_1.t_ - meas_0.t_);
        meas_1.t_ = t1_;
        meas_1.g_ = ((1.0 - c) * meas_0.g_ + c * meas_1.g_).eval();
        meas_1.a_ = ((1.0 - c) * meas_0.a_ + c * meas_1.a_).eval();
      }

      double delta_t = meas_1.t_ - meas_0.t_;

      // get average of gyro readings
      Eigen::Vector3d omega_true = (meas_0.g_ + meas_1.g_) / 2.0;
      omega_true -= b_g_hat;
      Eigen::Vector3d acc_true = (meas_0.a_ + meas_1.a_) / 2.0;
      acc_true -= b_a_hat;

      // save initial velocity and orientation estimates
      Eigen::Vector3d v_s_p0 = v_s_hat;
      Eigen::Quaterniond q_ws_p0 = q_ws_hat;

      QuaternionParameterization qp;

      Eigen::Matrix3d C_ws_hat = q_ws_hat.toRotationMatrix();
      Eigen::Matrix3d C_sw_hat = C_ws_hat.inverse();
          
      // propagate states using euler forward method
      Eigen::Quaterniond q_omega;
      q_omega.w() = 1.0;
      q_omega.vec() = 0.5 * omega_true * delta_t;
      Eigen::Matrix4d Omega = qp.oplus(q_omega);
      Eigen::Quaterniond q_ws_dot(Omega * q_ws_hat.coeffs());
          
      q_ws_hat = q_ws_hat * q_omega;
      q_ws_hat.normalize();
      Eigen::Vector3d g_w(0,0,params_.g_);
      v_s_hat = v_s_hat + ((acc_true + C_sw_hat * g_w - omega_true.cross(v_s_hat)) * delta_t);    
      b_a_hat = b_a_hat + ((-1.0/params_.b_a_tau_) * b_a_hat) * delta_t;
          
      // get average of velocity and orientation
      Eigen::Vector3d v_s_true = (v_s_p0);// + v_s_hat) / 2.0;
      Eigen::Quaterniond q_ws_true((q_ws_p0.coeffs() + q_ws_hat.coeffs()) / 2.0);
      q_ws_true.normalize();
      Eigen::Matrix3d C_ws_true = q_ws_p0.toRotationMatrix();
      Eigen::Matrix3d C_sw_true = C_ws_true.inverse();
          
      // calculate continuous time jacobian
      Eigen::Matrix<double,12,12> F_c = Eigen::Matrix<double,12,12>::Zero();
      F_c.block<3,3>(0,6) = -C_ws_true;
      Eigen::Matrix3d g_w_cross = Eigen::Matrix3d::Zero();
      g_w_cross(0,1) = params_.g_;
      g_w_cross(1,0) = -params_.g_;
      F_c.block<3,3>(3,0) = -q_ws_p0.toRotationMatrix().inverse() * g_w_cross;
      Eigen::Matrix3d omega_cross;
      omega_cross <<              0, -omega_true(2),  omega_true(1),
                      omega_true(2),              0, -omega_true(0),
                     -omega_true(1),  omega_true(0),              0;
      F_c.block<3,3>(3,3) = -omega_cross;
      Eigen::Matrix3d v_s_true_cross;
      v_s_true_cross <<           0, -v_s_true(2),  v_s_true(1),
                        v_s_true(2),            0, -v_s_true(0),
                       -v_s_true(1),  v_s_true(0),            0;
      F_c.block<3,3>(3,6) = -v_s_true_cross;
      F_c.block<3,3>(3,9) = -1.0 * Eigen::Matrix3d::Identity();
      F_c.block<3,3>(9,9) = (-1.0 / params_.b_a_tau_) * Eigen::Matrix3d::Identity();
      // approximate discrete time jacobian using Euler's method
      Eigen::Matrix<double,12,12> I_12 = Eigen::Matrix<double,12,12>::Identity();
      Eigen::Matrix<double,12,12> F_d;
      F_d = (I_12 + F_c * delta_t);
          
      F = F * F_d; // update total jacobian
      Eigen::Matrix<double,12,12> G = Eigen::Matrix<double,12,12>::Identity();
      G.block<3,3>(0,0) = C_ws_hat;

      // update covariance
      P = F_d * P * F_d.transpose().eval() + G * Q * G.transpose().eval() * delta_t;
    }
  }
  // finish jacobian
  Eigen::Matrix<double,12,12> F1 = Eigen::Matrix<double,12,12>::Identity();
  Eigen::Quaterniond q_ws_err = q_ws_hat * q_ws_1.inverse();
  q_ws_err.normalize();
  QuaternionParameterization qp;
  Eigen::Matrix4d q_err_oplus = qp.oplus(q_ws_err);
  F1.block<3,3>(0,0) = q_err_oplus.topLeftCorner<3,3>();
  Eigen::Matrix4d q_err_plus = qp.qplus(q_ws_err);
  F.block<3,3>(0,0) = F.block<3,3>(0,0) * q_err_plus.topLeftCorner<3,3>();
  //F = F * F1;
  F1 = -F1;
  // calculate residuals
  Eigen::Matrix<double,12,1> error;
  error.segment<3>(0) = 2.0 * q_ws_err.vec();
  error.segment<3>(3) = v_s_hat - v_s_1;
  error.segment<3>(6) = b_g_hat - b_g_1;
  error.segment<3>(9) = b_a_hat - b_a_1;

  // assign weighted residuals
  Eigen::Map<Eigen::Matrix<double,12,1> > weighted_error(residuals);
  P = 0.5 * P + 0.5 * P.transpose().eval();
  Eigen::Matrix<double,12,12> information = P.inverse();
  information = 0.5 * information + 0.5 * information.transpose().eval();

  Eigen::LLT<Eigen::Matrix<double,12,12>> lltOfInformation(information);
  Eigen::Matrix<double,12,12> square_root_information = lltOfInformation.matrixU();
  weighted_error = square_root_information * error;

  // get jacobians if requested
  if (jacobians != NULL)
  {
    // jacobian of residuals w.r.t. orientation at t0
    if (jacobians[0] != NULL)
    {
      // get minimal representation (3x12)
      Eigen::Matrix<double,12,3> J0_minimal;
      J0_minimal = square_root_information * F.block<12,3>(0,0);
          
      // get lift jacobian 
      // shifts minimal 3d representation to overparameterized 4d representation
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      QuaternionParameterization qp;
      qp.liftJacobian(parameters[0], J_lift.data());
      Eigen::Map<Eigen::Matrix<double,12,4,Eigen::RowMajor>> J0_mapped(jacobians[0]);
      J0_mapped = J0_minimal * J_lift;

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> 
            J0_minimal_mapped(jacobians_minimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
    // jacobian of residuals w.r.t. velocity at t0
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J1_mapped(jacobians[1]);
      J1_mapped = square_root_information * F.block<12,3>(0,3);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J1_minimal_mapped(jacobians_minimal[1]);
          J1_minimal_mapped = J1_mapped;
        }
      }
    }
    // jacobian of residuals w.r.t. gyro bias at t0
    if (jacobians[2] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J2_mapped(jacobians[2]);
      J2_mapped = square_root_information * F.block<12,3>(0,6);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[2] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J2_minimal_mapped(jacobians_minimal[2]);
          J2_minimal_mapped = J2_mapped;
        }
      }
    }
    // jacobian of residuals w.r.t. accel bias at t0
    if (jacobians[3] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J3_mapped(jacobians[3]);
      J3_mapped = square_root_information * F.block<12,3>(0,9);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[3] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J3_minimal_mapped(jacobians_minimal[3]);
          J3_minimal_mapped = J3_mapped;
        }
      }
    }
    // jacobian of residuals w.r.t. orientation at t1
    if (jacobians[4] != NULL)
    {
      // get minimal representation
      Eigen::Matrix<double,12,3> J4_minimal;
      J4_minimal = square_root_information * F1.block<12,3>(0,0);

      // get lift jacobian
      // shifts minimal 3d representation to overparameterized 4d representation
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      QuaternionParameterization qp;
      qp.liftJacobian(parameters[4], J_lift.data());

      Eigen::Map<Eigen::Matrix<double,12,4,Eigen::RowMajor>> J4_mapped(jacobians[4]);
      J4_mapped = J4_minimal * J_lift;

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[4] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J4_minimal_mapped(jacobians_minimal[4]);
          J4_minimal_mapped = J4_minimal;
        }
      }
    }
    // jacobian of residuals w.r.t. velocity at t1
    if (jacobians[5] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J5_mapped(jacobians[5]);
      J5_mapped = square_root_information * F1.block<12,3>(0,3);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[5] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J5_minimal_mapped(jacobians_minimal[5]);
          J5_minimal_mapped = J5_mapped;
        }
      }
    }
    // jacobian of residuals w.r.t. gyro bias at t1
    if (jacobians[6] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J6_mapped(jacobians[6]);
      J6_mapped = square_root_information * F1.block<12,3>(0,6);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[6] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J6_minimal_mapped(jacobians_minimal[6]);
          J6_minimal_mapped = J6_mapped;
        }
      }
    }
    // jacobian of residuals w.r.t. accel bias at t1
    if (jacobians[7] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>> J7_mapped(jacobians[7]);
      J7_mapped = square_root_information * F1.block<12,3>(0,9);

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[7] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,12,3,Eigen::RowMajor>>
            J7_minimal_mapped(jacobians_minimal[7]);
          J7_minimal_mapped = J7_mapped;
        }
      }
    }
  }
  return true;
}

size_t ImuVelocityCostFunction::ResidualDim() const
{
  return num_residuals();
}