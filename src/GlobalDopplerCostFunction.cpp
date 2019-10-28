#include <GlobalDopplerCostFunction.h>

GlobalDopplerCostFunction::GlobalDopplerCostFunction(double doppler,
      Eigen::Vector3d &target,
      Eigen::Matrix3d &radar_to_imu_mat,
      double weight) 
      : doppler_(doppler),
      target_ray_(target)
{
  set_num_residuals(1);
  mutable_parameter_block_sizes()->push_back(4);
  mutable_parameter_block_sizes()->push_back(3);
  target_ray_.normalize();

  // transform target ray from imu to radar board frame of reference
  Eigen::Vector3d radar_frame_ray = radar_to_imu_mat.transpose() * target_ray_;

  // reweight cost function based on x, y, and z components
  radar_frame_ray[0] *= 1.3;
  radar_frame_ray[1] *= 1.0;
  radar_frame_ray[2] *= 0.3;
  weight_ = weight * radar_frame_ray.lpNorm<1>();
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
  // get parameters
  const Eigen::Quaterniond q_WS(parameters[0][3],
                                parameters[0][0],
                                parameters[0][1],
                                parameters[0][2]);

  const Eigen::Vector3d v_W(parameters[1][0],
                            parameters[1][1],
                            parameters[1][2]);

  // get rotation matrices
  const Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
  const Eigen::Matrix3d C_SW = C_WS.transpose();

  // rotate velocity vector to sensor frame
  const Eigen::Vector3d v_S = C_SW * v_W;

  // get target velocity as -1.0 * body velocity
  Eigen::Vector3d v_target = -1.0 * v_S;

  // get projection of body velocity onto ray from target to sensor
  double v_projected = v_target.dot(target_ray_);

  // get residual as difference between v_r and doppler reading
  residuals[0] = (doppler_ - v_projected) * weight_;

  // calculate jacobian if required
  if (jacobians != NULL)
  {
    // jacobian w.r.t. orientation
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double,1,3> J0_minimal;
      J0_minimal = -v_W.cross(C_WS * target_ray_).transpose() * weight_;

      QuaternionParameterization qp;
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      qp.ComputeLiftJacobian(parameters[0],J_lift.data());

      Eigen::Map<Eigen::Matrix<double,1,4,Eigen::RowMajor>> 
        J0_mapped(jacobians[0]);
      J0_mapped = J0_minimal * J_lift;

      //LOG(ERROR) << "J0: " << J0_mapped;

      // assign minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> 
            J0_min_mapped(jacobians_minimal[0]);
          J0_min_mapped = J0_minimal;
        }
      }
    }
    // jacobian w.r.t. world frame velocity
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> 
        J1_mapped(jacobians[1]);
      J1_mapped = (C_WS * target_ray_).transpose() * weight_;

      //LOG(ERROR) << "J1: " << J1_mapped;

      // assign minimal jacobian if requested
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> 
            J1_min_mapped(jacobians_minimal[1]);
          J1_min_mapped = J1_mapped;
        }
      }
    }
  }
  //LOG(ERROR) << "residual: " << residuals[0];
  return true;
}

bool GlobalDopplerCostFunction::Evaluate(double const* const* parameters,
      double* residuals,
      double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}