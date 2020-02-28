#include <PointClusterCostFunction.h>

PointClusterCostFunction::PointClusterCostFunction(
  Eigen::Vector3d &target,
  double weight) 
  : target_(target),
    weight_(weight)
{
  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(7); // pose
  mutable_parameter_block_sizes()->push_back(4); // homogeneous point
}

PointClusterCostFunction::~PointClusterCostFunction(){}

size_t PointClusterCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool PointClusterCostFunction::EvaluateWithMinimalJacobians(
  double const* const* parameters,
  double* residuals,
  double** jacobians,
  double** jacobians_minimal) const
{
  return false;
}

bool PointClusterCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

