#include<AHRSOrientationCostFunction.h>

AHRSOrientationCostFunction::AHRSOrientationCostFunction(Eigen::Quaterniond &delta_q)
{

}

size_t AHRSOrientationCostFunction::ResidualDim() const
{

}

bool AHRSOrientationCostFunction::Evaluate(double const* const* parameters,
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
  
}