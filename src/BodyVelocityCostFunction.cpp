#include <BodyVelocityCostFunction.h>

BodyVelocityCostFunction::BodyVelocityCostFunction(double doppler,
      Eigen::Vector3d & target,
      double weight, double d) 
      : doppler_(doppler),
      target_ray_(target),
      weight_(weight),
      d_(d)
{
  set_num_residuals(4);
  mutable_parameter_block_sizes()->push_back(3); // body velocity
  mutable_parameter_block_sizes()->push_back(3); // target ray error
  target_ray_.normalize();
}

BodyVelocityCostFunction::~BodyVelocityCostFunction(){}

size_t BodyVelocityCostFunction::ResidualDim() const
{
  return num_residuals();
}

bool BodyVelocityCostFunction::EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const
{
  const Eigen::Vector3d v_body(parameters[0][0],
                               parameters[0][1],
                               parameters[0][2]);
  const Eigen::Vector3d delta(parameters[1][0],
                              parameters[1][1],
                              parameters[1][2]);

  // get target velocity as -1.0 * body velocity
  Eigen::Vector3d v_target = -1.0 * v_body;

  // get projection of body velocity onto ray from target to sensor
  double v_r = v_target.dot(target_ray_ + delta);
  // get residual as difference between v_r and doppler reading
  residuals[0] = (doppler_ - v_r) * weight_;
  residuals[1] = delta[0] * d_;
  residuals[2] = delta[1] * d_;
  residuals[3] = delta[2] * d_;

  // calculate jacobian if required
  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      // aren't linear functions just the best?
      Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J0(jacobians[0]);
      J0.setZero();
      J0.block<1,3>(0,0) = (target_ray_ + delta) * weight_;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J0_min(jacobians_minimal[0]);
          J0_min = J0;
        }
      }
    }
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J1(jacobians[1]);
      J1.block<1,3>(0,0).setZero();// = delta * weight_ * d_;
      J1.block<3,3>(1,0).setIdentity();
      J1.block<3,3>(1,0) *= d_;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J1_min(jacobians_minimal[1]);
          J1_min = J1;
        }
      }
    }
  }
  return true;
}

bool BodyVelocityCostFunction::Evaluate(double const* const* parameters,
      double* residuals,
      double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}