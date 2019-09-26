#include <VelocityMeasCostFunction.h>

VelocityMeasCostFunction::VelocityMeasCostFunction(
  Eigen::Vector3d &v_meas) : v_meas_(v_meas) 
{
  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(3);
}

VelocityMeasCostFunction::~VelocityMeasCostFunction(){}

bool VelocityMeasCostFunction::Evaluate(double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  Eigen::Map<const Eigen::Matrix<double,3,1>> v(&parameters[0][0]);
  Eigen::Map<Eigen::Matrix<double,3,1>> res(residuals);
  res = v - v_meas_;

  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J(jacobians[0]);
      J = Eigen::Matrix3d::Identity();
    }
  }
  return true;
}
