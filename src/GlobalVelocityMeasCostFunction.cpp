#include <GlobalVelocityMeasCostFunction.h>

GlobalVelocityMeasCostFunction::GlobalVelocityMeasCostFunction(
  Eigen::Vector3d &vicon_v,
  Eigen::Vector3d &radar_v) 
  : radar_v_(radar_v), 
    vicon_v_(vicon_v) 
{
  set_num_residuals(3);
  mutable_parameter_block_sizes()->push_back(4);
}

GlobalVelocityMeasCostFunction::~GlobalVelocityMeasCostFunction(){}

bool GlobalVelocityMeasCostFunction::Evaluate(double const* const* parameters,
  double* residuals,
  double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool GlobalVelocityMeasCostFunction::EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const
{
  Eigen::Quaterniond q_vr(parameters[0][3], // rotation from radar frame to vicon frame
                          parameters[0][0],
                          parameters[0][1],
                          parameters[0][2]);

  Eigen::Matrix3d C_vr = q_vr.toRotationMatrix();
  Eigen::Matrix3d C_rv = C_vr.transpose();

  Eigen::Map<Eigen::Matrix<double,3,1>> error(residuals);
  error = vicon_v_ - C_vr * radar_v_;

  //LOG(ERROR) << "rotation: " << q_vr.coeffs().transpose();
  //LOG(ERROR) << "vicon v: " << vicon_v_.transpose();
  //LOG(ERROR) << "radar v: " << radar_v_.transpose();
  //LOG(ERROR) << "residual: " << error.transpose() << "\n\n";

  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      QuaternionParameterization qp;
      Eigen::Matrix3d radar_v_cross;
      radar_v_cross <<         0.0, -radar_v_[2],  radar_v_[1],
                       radar_v_[2],          0.0, -radar_v_[0],
                      -radar_v_[1],  radar_v_[0],          0.0;
      Eigen::Matrix3d vicon_v_cross;
      vicon_v_cross <<         0.0, -vicon_v_[2],  vicon_v_[1],
                       vicon_v_[2],          0.0, -vicon_v_[0],
                      -vicon_v_[1],  vicon_v_[0],          0.0;

      Eigen::Matrix<double,3,3,Eigen::RowMajor> J_min = C_vr * radar_v_cross * C_rv;
      Eigen::Matrix<double,3,4,Eigen::RowMajor> J_lift;
      qp.ComputeLiftJacobian(parameters[0],J_lift.data());

      Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J(jacobians[0]);
      J = J_min * J_lift;

      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> 
            J_minimal(jacobians_minimal[0]);
          J_minimal = J_min;
        }
      }
    }
  }
  return true;
}

size_t GlobalVelocityMeasCostFunction::ResidualDim() const
{
  return num_residuals();
}
