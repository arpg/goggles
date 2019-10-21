#include<AHRSOrientationCostFunction.h>

AHRSOrientationCostFunction::AHRSOrientationCostFunction(
  Eigen::Quaterniond &delta_q,
  Eigen::Matrix3d &ahrs_to_imu)
{
  Eigen::Quaterniond ahrs_to_imu_quat(ahrs_to_imu);
  delta_q_ = ahrs_to_imu_quat * delta_q;

  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(4); // orientation at t0 (i,j,k,w)
  mutable_parameter_block_sizes()->push_back(4); // orientation at t1 (i,j,k,w)
}

AHRSOrientationCostFunction::~AHRSOrientationCostFunction() {}

size_t AHRSOrientationCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool AHRSOrientationCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool AHRSOrientationCostFunction::EvaluateWithMinimalJacobians(
  double const* const* parameters,
  double* residuals, 
  double** jacobians,
  double** jacobians_minimal) const
{
  Eigen::Quaterniond q_WS_0(parameters[0][3],
                        parameters[0][0],
                        parameters[0][1],
                        parameters[0][2]);
  Eigen::Quaterniond q_WS_1(parameters[1][3],
                        parameters[1][0],
                        parameters[1][1],
                        parameters[1][2]);
  /*
  Eigen::Matrix3d C_WS_0 = q_ws_0.toRotationMatrix();
  Eigen::Matrix3d C_SW_0 = C_WS_0.transpose();
  Eigen::Matrix3d C_WS_1 = q_ws_1.toRotationMatrix();
  Eigen::Matrix3d C_SW_1 = C_WS_1.transpose(); 
  */
  Eigen::Map<Eigen::Matrix<double,3,1,Eigen::RowMajor>> error(residuals);
  error = 2.0 * (delta_q_ * (q_WS_1.inverse() * q_WS_0)).vec();

  QuaternionParameterization qp;

  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      // get minimal jacobian
      Eigen::Matrix3d J0_minimal = (qp.oplus(delta_q_*q_WS_1.inverse())
                                    * qp.qplus(q_WS_0)).topLeftCorner(3,3);

      // get lift jacobian 
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      qp.ComputeLiftJacobian(parameters[0], J_lift.data());

      // move from minimal to overparameterized space
      Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0_mapped;
      J0_mapped = J0_minimal * J_lift;

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0_minimal_mapped;
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
  }
  if (jacobians[1] != NULL)
  {
    // get minimal jacobian
    Eigen::Matrix3d J1_minimal = -(qp.oplus(delta_q_) 
      * qp.qplus(q_WS_0) 
      * qp.oplus(q_WS_1.inverse())).topLeftCorner(3,3);

    // get lift jacobian
    Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
    qp.ComputeLiftJacobian(parameters[1], J_lift.data());

    // move from minimal to overparameterized space
    Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1_mapped;
    J1_mapped = J0_minimal * J_lift;

    // get minimal jacobian if requested
    if (jacobians_minimal != NULL)
    {
      if (jacobians_minimal[1] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1_minimal_mapped;
        J1_minimal_mapped = J1_minimal;
      }
    }
  }
}