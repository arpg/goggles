#include "glog/logging.h"
#include "include/CeresCostFunctions.h"

double sinc_test(double x)
{
  if(fabs(x)>1e-10) 
  {
    return sin(x)/x;
  }
  else
  {
    static const double c_2=1.0/6.0;
    static const double c_4=1.0/120.0;
    static const double c_6=1.0/5040.0; 
    const double x_2 = x*x;
    const double x_4 = x_2*x_2;
    const double x_6 = x_2*x_2*x_2;
    return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
  }
}


TEST(goggleTests, ImuIntegration)
{
  // set the imu parameters
  ImuParams imuParameters;
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.sigma_g_ = 6.0e-4;
  imuParameters.sigma_a_ = 2.0e-3;
  imuParameters.sigma_b_g_ = 3.0e-6;
  imuParameters.sigma_b_a_ = 2.0e-5;
  imuParameters.b_a_tau_ = 3600.0;

  double imu_rate = 1000.0;

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1,10.0);
  const double w_a_W_y = Eigen::internal::random(0.1,10.0);
  const double w_a_W_z = Eigen::internal::random(0.1,10.0);
  const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);

  const double duration = 1.0
  const double dt = 1.0 / imu_rate;
  std::vector<ImuMeasurement> imuMeasurements;

  // states
  Eigen::Quaterniond q_ws;
  Eigen::Vector3d v_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_g = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();

  // starting state
  Eigen::Quaterniond q_ws_0;
  Eigen::Vector3d v_s_0;
  Eigen::Vector3d b_g_0;
  Eigen::Vector3d b_a_0;
  double t0;

  // ending state
  Eigen::Quaterniond q_ws_1
  Eigen::Vector3d v_s_1;
  Eigen::Vector3d b_g_1;
  Eigen::Vector3d b_a_1;
  double t1;

  // generate IMU measurements without noise
  for (int i = 0; i < int(duration*imu_rate); i++)
  {
    if (i == 10) // set as starting pose
    {
      q_ws_0 = q_ws;
      v_s_0 = v_s;
      b_g_0 = b_g;
      b_a_0 = b_a;
      t0 = time;
    }
    if (i == int(duration*imu_rate) - 10) // set as end pose
    {
      q_ws_1 = q_ws;
      v_s_1 = v_s;
      b_g_1 = b_g;
      b_a_1 = b_a;
      t1 = time;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                    m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                    m_omega_S_z*sin(omega_S_z*time+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                    m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                    m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

    Eigen::Quaterniond dq;

    // propagate orientation
    const double theta_half = omega_S.norm()*dt*0.5;
    const double sinc_theta_half = sinc_test(theta_half); 
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half*0.5*dt*omega_S;
    dq.w() = cos_theta_half;
    q_ws = q_ws * dq;

    // propagate speed
    v_w += dt*a_W; 
    v_s = q_ws.inverse().toRotationMatrix() * v_w;

    // generate measurements (with no noise)
    // TO DO: add biases
    ImuMeasurement new_meas;
    new_meas.t_ = time;
    new_meas.g_ omega_S;
    new_meas.a_ = q_ws.inverse().toRotationMatrix() * (a_W + Eigen::Vector3d(0,0,imuParameters.g_));

    imuMeasurements.push_back(new_meas);
  }

  // propagate states from t0 to t1 using imu measurements
  for (int i = 0; i < imuMeasurements.size(); i++)
  {
    if (imuMeasurements[i].t_ + dt > t0 && imuMeasurements[i].t_ - dt < t1)
    {
      
    }
  }

  // compare groundtruth states at t1 to states at t1 from imu integration
}