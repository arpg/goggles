#include<HomogeneousPointParameterization.h>

HomogeneousPointParameterization::HomogeneousPointParameterization(){}
HomogeneousPointParameterization::~HomogeneousPointParameterization(){}

bool HomogeneousPointParameterization::Plus(const double* x, 
                                            const double* delta,
                                            double* x_plus_delta) const
{
  return plus(x, delta, x_plus_delta);
}

bool HomogeneousPointParameterization::plus(const double* x, 
                                            const double* delta,
                                            double* x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> delta_(delta);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<Eigen::Vector4d> x_plus_delta_(x_plus_delta);

  x_plus_delta_ = x_ + Eigen::Vector4d(delta_[0], delta_[1], delta_[2], 0.0);

  return true;
}

bool HomogeneousPointParameterization::Minus(const double* x,
                                             const double* x_plus_delta,
                                             double* delta) const
{
  return minus(x, x_plus_delta, delta);
}

bool HomogeneousPointParameterization::minus(const double* x,
                                             const double* x_plus_delta,
                                             double* delta) const
{
  Eigen::Map<Eigen::Vector3d> delta_(delta);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<const Eigen::Vector4d> x_plus_delta_(x_plus_delta);

  delta_ = (x_plus_delta_ - x_).head<3>();

  return true;
}

bool HomogeneousPointParameterization::ComputeLiftJacobian(
  const double* x, double* jacobian) const
{
  return liftJacobian(x, jacobian);
}

bool HomogeneousPointParameterization::ComputeJacobian(
  const double* x, double* jacobian) const
{
  return plusJacobian(x, jacobian);
}

bool HomogeneousPointParameterization::plusJacobian(const double* x, 
                                                    double* jacobian) const
{
  Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> Jp(jacobian);
  Jp.setZero();
  Jp.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity();

  return true;
}

bool HomogeneousPointParameterization::liftJacobian(
  const double* x, double* jacobian) const
{
  Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> Jp(jacobian);
  Jp.setZero();
  Jp.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity();

  return true;
}