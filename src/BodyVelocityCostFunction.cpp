#include <BodyVelocityCostFunction.h>

BodyVelocityCostFunction::BodyVelocityCostFunction(double doppler,
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

BodyVelocityCostFunction::~BodyVelocityCostFunction(){}

bool BodyVelocityCostFunction::Evaluate(double const* const* parameters,
      double* residuals,
      double** jacobians) const
{
      //LOG(ERROR) << "evaluating";
  Eigen::Map<const Eigen::Vector3d> v_body(&parameters[0][0]);
      //LOG(ERROR) << "v_body: " << v_body.transpose();
      // get target velocity as -1.0 * body velocity
  Eigen::Vector3d v_target = -1.0 * v_body;
      //LOG(ERROR) << "v_target: " << v_target.transpose();
      // get projection of body velocity onto ray from target to sensor
  double v_r = v_target.dot(target_ray_);
      //LOG(ERROR) << "target_ray: " << target_ray_.transpose();
      //LOG(ERROR) << "v_r: " << v_r;
      // get residual as difference between v_r and doppler reading
  residuals[0] = (doppler_ - v_r) * weight_;
      //LOG(ERROR) << "weight: " << weight_;
      //LOG(ERROR) << "residual: "  << residuals[0];
      // calculate jacobian if required
  if (jacobians != NULL)
  {
        //LOG(ERROR) << "evaluating jacobians";
        // aren't linear functions just the best?
    jacobians[0][0] = target_ray_[0] * weight_;
    jacobians[0][1] = target_ray_[1] * weight_;
    jacobians[0][2] = target_ray_[2] * weight_;
        //LOG(ERROR) << "jacobians done";
  }
      //LOG(ERROR) << "done";
  return true;
}