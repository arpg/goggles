#include <GlobalDopplerCostFunction.h>

GlobalDopplerCostFunction::GlobalDopplerCostFunction(double doppler,
      Eigen::Vector3d & target,
      double weight) 
      : doppler_(doppler),
      target_ray_(target),
      weight_(weight) 
{
  set_num_residuals(1);
  mutable_parameter_block_sizes()->push_back(3);
  target_ray_.normalize();
}

GlobalDopplerCostFunction::~GlobalDopplerCostFunction(){}

size_t GlobalDopplerCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool GlobalDopplerCostFunction::EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const
{
  Eigen::Map<const Eigen::Vector3d> v_body(&parameters[0][0]);

  // get target velocity as -1.0 * body velocity
  Eigen::Vector3d v_target = -1.0 * v_body;

  // get projection of body velocity onto ray from target to sensor
  double v_r = v_target.dot(target_ray_);

  // get residual as difference between v_r and doppler reading
  residuals[0] = (doppler_ - v_r) * weight_;

  // calculate jacobian if required
  if (jacobians != NULL)
  {
    // aren't linear functions just the best?
    jacobians[0][0] = target_ray_[0] * weight_;
    jacobians[0][1] = target_ray_[1] * weight_;
    jacobians[0][2] = target_ray_[2] * weight_;

    if (jacobians_minimal != NULL)
    {
      jacobians_minimal[0][0] = jacobians[0][0];
      jacobians_minimal[0][1] = jacobians[0][1];
      jacobians_minimal[0][2] = jacobians[0][2];
    }
  }
  return true;
}

bool GlobalDopplerCostFunction::Evaluate(double const* const* parameters,
      double* residuals,
      double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}