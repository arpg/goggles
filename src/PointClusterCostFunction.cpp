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
  // get parameters
  const Eigen::Vector3d r_WS(parameters[0][0],
                             parameters[0][1],
                             parameters[0][2]);

  const Eigen::Quaterniond q_WS(parameters[0][6],
                                parameters[0][3],
                                parameters[0][4],
                                parameters[0][5]);

  const Transformation T_WS(r_WS,q_WS);

  const Eigen::Vector4d h_p(parameters[1][0],
                            parameters[1][1],
                            parameters[1][2],
                            parameters[1][3]);

  // get transformation matrix
  Eigen::Matrix4d T_WS_mat = T_WS.T();

  // transform target from sensor frame to global frame
  Eigen::Vector4d target_s = Eigen::Vector4d(target_[0],
                                             target_[1],
                                             target_[2],
                                             1.0);
  Eigen::Vector4d target_w = T_WS_mat * target_s;

  // get residual as the distance between the target and the landmark
  residuals[0] = weight_ * (target_w[0] - h_p[0]);
  residuals[1] = weight_ * (target_w[1] - h_p[1]);
  residuals[2] = weight_ * (target_w[2] - h_p[2]);

  if (jacobians != NULL)
  {
    // jacobian w.r.t. pose
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double,3,6> J0_minimal;
      J0_minimal.setZero();

      PoseParameterization p;
      J0_minimal.topLeftCorner(3,3) = Eigen::Matrix3d::Identity();
      Eigen::Matrix3d p_cross;
      tw_cross << 0,          -target_w[2], target_w[1],
                  target_w[2], 0,          -target_w[0],
                 -target_w[1], target_w[0], 0;
      J0_minimal.topRightCorner(3,3) = -tw_cross;
      J0_minimal *= weight_;

      Eigen::Matrix<double,6,7,Eigen::RowMajor> J_lift;
      p.ComputeLiftJacobian(parameters[0], J_lift.data());

      Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor>> 
        J0_mapped(jacobians[0]); 
      J0_mapped = J0_minimal * J_lift;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor>>
            J0_min_mapped(jacobians_minimal[0]);
          J0_min_mapped = J0_minimal;
        }
      }
    }
    // jacobian w.r.t. landmark location
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> 
        J1_mapped(jacobians[1]);

      J1_mapped.setZero();
      J1_mapped.topLeftCorner(3,3) = Eigen::Matrix3d::Identity();
      J1_mapped *= -weight_;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>
            J1_minimal_mapped(jacobians_minimal[1]);
          J1_minimal_mapped = -weight_ * Eigen::Matrix3d::Identity();
        }
      }
    }
  }

  return true;
}

bool PointClusterCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

