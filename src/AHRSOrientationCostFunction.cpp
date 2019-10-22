#include<AHRSOrientationCostFunction.h>

AHRSOrientationCostFunction::AHRSOrientationCostFunction()
{
  q_WS_meas_ = Eigen::Quaterniond::Identity();
}

AHRSOrientationCostFunction::AHRSOrientationCostFunction(
  Eigen::Quaterniond &q_WS_meas_ahrs,
  Eigen::Matrix3d &ahrs_to_imu,
  Eigen::Matrix3d &initial_orientation,
  double weight)
{
  Eigen::Matrix3d C_WS_meas_ahrs_mat = q_WS_meas_ahrs.toRotationMatrix();

  Eigen::Matrix3d C_WS_meas_mat = ahrs_to_imu * 
                                  C_WS_meas_ahrs_mat * 
                                  initial_orientation;
  q_WS_meas_ = Eigen::Quaterniond(C_WS_meas_mat);

  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(4); // orientation at t0 (i,j,k,w)
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
  Eigen::Quaterniond q_WS(parameters[0][3],
                        parameters[0][0],
                        parameters[0][1],
                        parameters[0][2]);

  Eigen::Map<Eigen::Vector3d> error(residuals);
  error = 2.0 * (q_WS.inverse() * q_WS_meas_).vec();
  //error = 2.0 * (delta_q_.inverse() * (q_WS_1.inverse() * q_WS_0)).vec();

  QuaternionParameterization qp;

  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      // get minimal jacobian
      Eigen::Matrix3d J0_minimal = -0.5 * (qp.oplus(q_WS.inverse())
                                    * qp.qplus(q_WS_meas_)).topLeftCorner(3,3);

      // get lift jacobian 
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      qp.ComputeLiftJacobian(parameters[0], J_lift.data());

      // move from minimal to overparameterized space
      Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J0_mapped(jacobians[0]);
      J0_mapped = J0_minimal * J_lift;

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0_minimal_mapped(
            jacobians_minimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
  }
}