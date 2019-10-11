#include <ImuIntegrator.h>

ImuIntegrator::ImuIntegrator(ImuMeasurement &m0, 
  ImuMeasurement &m1, double g_w, double tau)
: m0_(m0), m1_(m1), tau_(tau)
{
  g_w_ << 0,0,g_w; // define world gravity as in positive Z direction
}

ImuIntegrator::~ImuIntegrator(){}
    
void ImuIntegrator::operator() (const std::vector<double> &x, 
                                std::vector<double> &dxdt, 
                                const double t)
{
  // map in and out states to Eigen datatypes for convenience
  Eigen::Map<const Eigen::Quaterniond> q_ws(&x[0]); // sensor-to-world rotation
  Eigen::Map<const Eigen::Vector3d> v_s(&x[4]); // sensor frame velocity
  Eigen::Map<const Eigen::Vector3d> b_g(&x[7]); // gyro biases
  Eigen::Map<const Eigen::Vector3d> b_a(&x[10]); // accelerometer biases
  Eigen::Map<Eigen::Quaterniond> q_ws_dot(&dxdt[0]); 
  Eigen::Map<Eigen::Vector3d> v_s_dot(&dxdt[4]);
  Eigen::Map<Eigen::Vector3d> b_g_dot(&dxdt[7]);
  Eigen::Map<Eigen::Vector3d> b_a_dot(&dxdt[10]);

  // get interpolated imu measurement at time t
  double t0 = m0_.t_;
  double t1 = m1_.t_;
  double t_span = t1 - t0;
  double c = 0.5;//(t - t0) / t_span;
  Eigen::Vector3d g = ((1.0 - c) * m0_.g_ + c * m1_.g_).eval();
  Eigen::Vector3d a = ((1.0 - c) * m0_.a_ + c * m1_.a_).eval();

  g = g - b_g; // subtract gyro biases
  a = a - b_a; // subtract accel biases

  // define differential equations
  // ref: Leutenegger et al, 2015
  QuaternionParameterization qp;
      
  Eigen::Matrix4d Omega;
  Eigen::Quaterniond omega_quat(1, -g(0), -g(1), -g(2));
  Omega = qp.oplus(omega_quat);
  q_ws_dot.coeffs() = 0.5 * Omega * q_ws.coeffs();
  q_ws_dot.normalize();

  Eigen::Matrix3d C_sw = q_ws.toRotationMatrix().inverse();
  v_s_dot = a + (C_sw * g_w_) - g.cross(v_s);
  b_g_dot = Eigen::Vector3d::Zero();
  b_a_dot = -(1.0 / tau_) * b_a;  
}
