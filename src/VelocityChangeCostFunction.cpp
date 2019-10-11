#include <VelocityChangeCostFunction.h>

VelocityChangeCostFunction::VelocityChangeCostFunction(double t) : delta_t_(t) 
{
  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(3);
  mutable_parameter_block_sizes()->push_back(3);
}

VelocityChangeCostFunction::~VelocityChangeCostFunction(){}

bool VelocityChangeCostFunction::Evaluate(double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

size_t VelocityChangeCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool VelocityChangeCostFunction::EvaluateWithMinimalJacobians(
    double const* const* parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const
{
  Eigen::Map<const Eigen::Vector3d> v0(&parameters[0][0]);
  Eigen::Map<const Eigen::Vector3d> v1(&parameters[1][0]);
  Eigen::Vector3d res;
  res = (v0 - v1) / delta_t_;
  residuals[0] = res[0];
  residuals[1] = res[1];
  residuals[2] = res[2];
  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0(jacobians[0]);
      J0 = Eigen::Matrix3d::Identity() *(1.0 / delta_t_);

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> 
            J0_minimal(jacobians_minimal[0]);
          J0_minimal = J0;
        }
      }
    }
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1(jacobians[1]);
      J1 = Eigen::Matrix3d::Identity() * (1.0 / delta_t_) * -1.0;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>
            J1_minimal(jacobians_minimal[1]);
          J1_minimal = J1;
        }
      }
    }
  }
  return true;
}