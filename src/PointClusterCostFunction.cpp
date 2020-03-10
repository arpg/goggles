#include <PointClusterCostFunction.h>

PointClusterCostFunction::PointClusterCostFunction(
  Eigen::Vector3d &target) 
{
  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(7); // pose
  mutable_parameter_block_sizes()->push_back(4); // homogeneous point

  // calculate weight
  
  /*
  //// CARTESIAN APPROXIMATION ////
  target_ = target;
  Eigen::Vector3d unit_ray = target_.normalized().cwiseAbs();
  
  weight_ = Eigen::Matrix3d::Identity();
  weight_(1,1) = 1.0 - unit_ray[1];
  weight_(2,2) = 1.0 - unit_ray[2];
  */

  //// POLAR FORMULATION ////
  target_[0] = target.norm(); // range
  target_[1] = atan2(target[1],target[0]); // azimuth
  target_[2] = atan2(target[2],target.head(2).norm()); // elevation

  weight_ = Eigen::Matrix3d::Identity();
  weight_(1,1) = cos(target_[1]);
  weight_(2,2) = cos(target_[2]);
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
  Eigen::Matrix4d T_SW_mat = T_WS.inverse().T();

  Eigen::Vector4d hp_s = T_SW_mat * h_p;

  //// POLAR FORMULATION ////
  Eigen::Vector3d point_s = hp_s.head(3) / hp_s[3];
  double range = point_s.norm();
  double azimuth = atan2(point_s[1],point_s[0]); 
  double elevation = atan2(point_s[2],point_s.head(2).norm());

  Eigen::Vector3d error;
  error[0] = range - target_[0];
  error[1] = azimuth - target_[1];
  error[2] = elevation - target_[2];

  Eigen::Map<Eigen::Vector3d> weighted_error(residuals);
  weighted_error = error;

  // Get jacobians if requested
  if (jacobians != NULL)
  {
    // jacobian w.r.t. pose
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double,3,6> J0_minimal;
      J0_minimal.setZero();
      PoseParameterization p;

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
    // jacobian w.r.t. scatterer location
    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> 
        J1_mapped(jacobians[1]);

      J1_mapped.setZero();

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>
            J1_minimal_mapped(jacobians_minimal[1]);
          J1_minimal_mapped.setZero();
        }
      }
    }
  }

  /*
  //// CARTESIAN APPROXIMATION ////
  // get residual as the distance between the target and the landmark

  // transform target from sensor frame to global frame
  Eigen::Vector4d target_s = Eigen::Vector4d(target_[0],
                                             target_[1],
                                             target_[2],
                                             1.0);
  Eigen::Vector4d target_w = T_WS_mat * target_s;
  Eigen::Vector3d error;
  error = (target_s.head(3) / target_s[3]) - (hp_s.head(3) / hp_s[3]);

  Eigen::Map<Eigen::Vector3d> weighted_error(residuals);
  weighted_error = weight_ * error;

  if (jacobians != NULL)
  {
    // jacobian w.r.t. pose
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double,3,6> J0_minimal;
      J0_minimal.setZero();

      PoseParameterization p;
      J0_minimal.topLeftCorner(3,3) = weight_ * T_WS.inverse().C();

      Eigen::Matrix3d hp_cross;
      Eigen::Vector3d hp_point = h_p.head(3) / h_p[3];

      hp_cross << 0, -hp_point[2], hp_point[1],
                  hp_point[2], 0, -hp_point[0],
                  -hp_point[1], hp_point[0], 0;
      J0_minimal.topRightCorner(3,3) = -weight_ * T_WS.inverse().C() * hp_cross;

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
      J1_mapped.topLeftCorner(3,3) = -weight_ * T_WS.inverse().C();

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>
            J1_minimal_mapped(jacobians_minimal[1]);
          J1_minimal_mapped = -weight_ * T_WS.inverse().C();
        }
      }
    }
  }
  */
  return true;
}

bool PointClusterCostFunction::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

