#include<AHRSYawCostFunction.h>

AHRSYawCostFunction::AHRSYawCostFunction()
{
  q_WS_meas_ = Eigen::Quaterniond::Identity();
}

AHRSYawCostFunction::AHRSYawCostFunction(
  Eigen::Quaterniond &q_WS_meas,
  bool invert)
{
  q_WS_meas_ = q_WS_meas;
  q_WS_meas_.x() = 0.0; // zero pitch
  q_WS_meas_.y() = 0.0; // zero roll
  q_WS_meas_.normalize();

  if (invert)
    q_WS_meas_ = q_WS_meas_.inverse();

  set_num_residuals(1);
  mutable_parameter_block_sizes()->push_back(4); // orientation (i,j,k,w)
}

AHRSYawCostFunction::~AHRSYawCostFunction() {}

size_t AHRSYawCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool AHRSYawCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool AHRSYawCostFunction::EvaluateWithMinimalJacobians(
  double const* const* parameters,
  double* residuals, 
  double** jacobians,
  double** jacobians_minimal) const
{
  Eigen::Quaterniond q_WS(parameters[0][3],
                        parameters[0][0],
                        parameters[0][1],
                        parameters[0][2]);

  // zero pitch and roll
  q_WS.x() = 0.0;
  q_WS.y() = 0.0;
  q_WS.normalize();

  Eigen::Vector3d error;
  error = 2.0 * (q_WS.inverse() * q_WS_meas_).vec();
  residuals[0] = error[2];

  QuaternionParameterization qp;

  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      // get minimal jacobian
      Eigen::Matrix<double,1,3> J0_minimal = -(qp.oplus(q_WS.inverse())
                                    * qp.qplus(q_WS_meas_)).block<1,3>(2,0);

      // get lift jacobian 
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      qp.ComputeLiftJacobian(parameters[0], J_lift.data());

      // move from minimal to overparameterized space
      Eigen::Map<Eigen::Matrix<double,1,4,Eigen::RowMajor>> J0_mapped(jacobians[0]);
      J0_mapped = J0_minimal * J_lift;

      // get minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> J0_minimal_mapped(
            jacobians_minimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
  }
}