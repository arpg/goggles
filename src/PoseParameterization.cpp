#include <PoseParameterization.h>

PoseParameterization::PoseParameterization(){}

PoseParameterization::~PoseParameterization(){}

bool PoseParameterization::Plus(const double* x, 
                                const double* delta, 
                                double* x_plus_delta) const
{
  return plus(x, delta, x_plus_delta);
}

bool PoseParameterization::plus(const double* x, 
                                const double* delta, 
                                double* x_plus_delta) const
{
  Eigen::Map<const Eigen::Matrix<double,6,1>> delta_(delta);
  Transformation T(Eigen::Vector3d(x[0],x[1],x[2]),
    Eigen::Quaterniond(x[6],x[3],x[4],x[5]));

  T.oplus(delta_);

  const Eigen::Vector3d r = T.r();
  const Eigen::Vector4d q = T.q().coeffs();
  x_plus_delta[0] = r[0];
  x_plus_delta[1] = r[1];
  x_plus_delta[2] = r[2];
  x_plus_delta[3] = q[0];
  x_plus_delta[4] = q[1];
  x_plus_delta[5] = q[2];
  x_plus_delta[6] = q[3];

  return true;
}

bool PoseParameterization::Minus(const double* x,
                                 const double* x_plus_delta,
                                 double* delta) const
{
  return minus(x, x_plus_delta, delta);
}

bool PoseParameterization::minus(const double* x,
                                 const double* x_plus_delta,
                                 double* delta) const
{
  delta[0] = x_plus_delta[0] - x[0];
  delta[1] = x_plus_delta[1] - x[1];
  delta[2] = x_plus_delta[2] - x[2];
  const Eigen::Quaterniond q_plus_delta_(x_plus_delta[6], x_plus_delta[3],
                                         x_plus_delta[4], x_plus_delta[5]);
  const Eigen::Quaterniond q_(x[6], x[3], x[4], x[5]);
  Eigen::Map<Eigen::Vector3d> delta_q_(&delta[3]);
  delta_q_ = 2.0 * (q_plus_delta_ * q_.inverse()).coeffs().template head<3>(0);
  return true;
}

bool PoseParameterization::ComputeLiftJacobian(const double* x,
                                               double* jacobian) const
{
  return liftJacobian(x, jacobian);
}

bool PoseParameterization::plusJacobian(const double* x,
                                        double* jacobian) const
{
  Eigen::Map<Eigen::Matrix<double,7,6,Eigen::RowMajor>> Jp(jacobian);
  Transformation T(Eigen::Vector3d(x[0],x[1],x[2]),
                   Eigen::Quaterniond(x[6],x[3],x[4],x[5]));
  T.oplusJacobian(Jp);
  return true;
}

bool PoseParameterization::liftJacobian(const double* x,
                                        double* jacobian) const
{
  Eigen::Map<Eigen::Matrix<double,6,7,Eigen::RowMajor>> J_lift(jacobian);
  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  J_lift.setZero();
  J_lift.topLeftCorner<3,3>().setIdentity();
  Eigen::Matrix4d q_plus = qplus(q_inv);
  Eigen::Matrix<double,3,4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3,1>().setZero();
  Jq_pinv.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 2.0;
  J_lift.bottomRightCorner<3,4>() = Jq_pinv * q_plus;

  return true;
}

bool PoseParameterization::ComputeJacobian(const double* x,
                                           double* jacobian) const
{
  return plusJacobian(x, jacobian);
}

bool PoseParameterization::VerifyJacobianNumDiff(const double* x,
                                                 double* jacobian,
                                                 double* jacobianNumDiff) const
{
  plusJacobian(x, jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jpn(
      jacobianNumDiff);
  double dx = 1e-9;
  Eigen::Matrix<double, 7, 1> xp;
  Eigen::Matrix<double, 7, 1> xm;
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta;
    delta.setZero();
    delta[i] = dx;
    Plus(x, delta.data(), xp.data());
    delta[i] = -dx;
    Plus(x, delta.data(), xm.data());
    Jpn.col(i) = (xp - xm) / (2 * dx);
  }
  if ((Jp - Jpn).norm() < 1e-6)
    return true;
  else
    return false;
}

bool PoseParameterization::Verify(const double* x_raw, 
                             double perturbation_magnitude) const
{
  const ceres::LocalParameterization* casted = 
    dynamic_cast<const ceres::LocalParameterization*>(this);

  if (!casted) return false;

  LOG(INFO) << "verifying plus and minus";
  Eigen::VectorXd x(casted->GlobalSize());
  memcpy(x.data(), x_raw, sizeof(double) * casted->GlobalSize());
  Eigen::VectorXd delta_x(casted->LocalSize());
  Eigen::VectorXd x_plus_delta(casted->GlobalSize());
  Eigen::VectorXd delta_x2(casted->LocalSize());
  delta_x.setRandom();
  delta_x *= perturbation_magnitude;
  casted->Plus(x.data(), delta_x.data(), x_plus_delta.data());
  this->Minus(x.data(), x_plus_delta.data(), delta_x2.data());

  if ((delta_x2 - delta_x).norm() > 1.0e-12) return false;

      // verify plusJacobian through numDiff
  LOG(INFO) << "verifying plusJacobian";
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J_plus_num_diff(
    casted->GlobalSize(), casted->LocalSize());
  const double dx = 1.0e-9;
  for (int i = 0; i < casted->LocalSize(); i++)
  {
    Eigen::VectorXd delta_p(casted->LocalSize());
    delta_p.setZero();
    delta_p[i] = dx;
    Eigen::VectorXd delta_m(casted->LocalSize());
    delta_m.setZero();
    delta_m[i] = -dx;

    Eigen::VectorXd x_p(casted->GlobalSize());
    Eigen::VectorXd x_m(casted->GlobalSize());
    memcpy(x_p.data(), x_raw, sizeof(double)*casted->GlobalSize());
    memcpy(x_m.data(), x_raw, sizeof(double)*casted->GlobalSize());
    casted->Plus(x.data(), delta_p.data(), x_p.data());
    casted->Plus(x.data(), delta_m.data(), x_m.data());
    J_plus_num_diff.col(i) = (x_p - x_m) / (2.0 * dx);
  }

      // verify lift
  LOG(INFO) << "verifying lift";
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J_plus(casted->GlobalSize(),
    casted->LocalSize());
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J_lift(casted->LocalSize(),
    casted->GlobalSize());

  casted->ComputeJacobian(x_raw, J_plus.data());
  ComputeLiftJacobian(x_raw, J_lift.data());
  Eigen::MatrixXd identity(casted->LocalSize(), casted->LocalSize());
  identity.setIdentity();

  if (((J_lift * J_plus) - identity).norm() > 1.0e-6) return false;
  if ((J_plus - J_plus_num_diff).norm() > 1.0e-6) return false;

  return true;
}

Eigen::Matrix4d PoseParameterization::oplus(
    const Eigen::Quaterniond &q_BC) const
{
  Eigen::Vector4d q = q_BC.coeffs();
  Eigen::Matrix4d Q;

  Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
  Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
  Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];

  return Q;
}

Eigen::Matrix4d PoseParameterization::qplus(
    const Eigen::Quaterniond &q_BC) const
{
  Eigen::Vector4d q = q_BC.coeffs();
  Eigen::Matrix4d Q;

  Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
  Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
  Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];

  return Q;
}