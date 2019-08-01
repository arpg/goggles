#include <boost/numeric/odeint.hpp>

struct ImuMeasurement
{
	Eigen::Vector3d g_; // gyro reading
	Eigen::Vector3d a_; // accelerometer reading
	double t_;          // timestamp
};

struct ImuParams
{
	double g_; // gravity magnitude
	double sigma_g_; // gyro noise prior
	double sigma_a_; // accelerometer noise prior
	double sigma_b_g_; // gyro bias noise prior
	double sigma_b_a_; // accelerometer bias noise prior
	double b_a_tau_; // accelerometer bias random walk parameter
};

class BodyVelocityCostFunction : public ceres::CostFunction
{
  public:
    BodyVelocityCostFunction(double doppler,
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

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const
    {
      Eigen::Map<const Eigen::Vector3d> v_body(&parameters[0][0]);

      // get target velocity as -1.0 * body velocity
      Eigen::Vector3d v_target = -1.0 * v_body;

      // get projection of body velocity onto ray from target to sensor
      double v_r = v_target.dot(target_ray_);

      // get residual as difference between v_r and doppler reading
      residuals[0] = (doppler_ - v_r) * weight_;

      // calculate jacobian if required
      if (jacobians != NULL)
      {
        // aren't linear functions just the best?
        jacobians[0][0] = target_ray_[0] * weight_;
        jacobians[0][1] = target_ray_[1] * weight_;
        jacobians[0][2] = target_ray_[2] * weight_;
      }
      return true;
    }

  protected:
    double doppler_;
    double weight_;
    Eigen::Vector3d target_ray_;
};

class VelocityChangeCostFunction : public ceres::CostFunction
{
  public:

    VelocityChangeCostFunction(double t) : delta_t_(t) 
    {
      set_num_residuals(3);
      mutable_parameter_block_sizes()->push_back(3);
      mutable_parameter_block_sizes()->push_back(3);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const
    {
      Eigen::Map<const Eigen::Vector3d> v0(&parameters[0][0]);
      Eigen::Map<const Eigen::Vector3d> v1(&parameters[1][0]);
      Eigen::Vector3d res;
      res = (v0 - v1) / delta_t_;
      residuals[0] = res[0];
      residuals[1] = res[1];
      residuals[2] = res[2];
      if (jacobians != NULL)
      {
        if (jacobians[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J0(jacobians[0]);
          J0 = Eigen::Matrix3d::Identity() *(1.0 / delta_t_);
        }
        if (jacobians[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J1(jacobians[1]);
          J1 = Eigen::Matrix3d::Identity() * (1.0 / delta_t_) * -1.0;
        }
      }
      return true;
    }

  protected:
    double delta_t_;
};

class ImuIntegrator
{
	public:
		ImuIntegrator(ImuMeasurement &m0, ImuMeasurement &m1, double g_w, double tau)
			: m0_(m0), m1_(m1), tau_(tau)
		{
			g_w_ << 0,0,g_w; // define world gravity as in positive Z direction
		}
		
		void operator() (const std::vector<double> &x, 
									std::vector<double> &dxdt, 
									const double t)
		{
			// map in and out states to Eigen datatypes for convenience
			Eigen::Map<const Eigen::Vector4d> q_ws(&x[0]); // sensor-to-world rotation
																										 // assumes i,j,k,w representation
			Eigen::Map<const Eigen::Vector3d> v_s(&x[4]); // sensor frame velocity
			Eigen::Map<const Eigen::Vector3d> b_g(&x[7]); // gyro biases
			Eigen::Map<const Eigen::Vector3d> b_a(&x[10]); // accelerometer biases
			Eigen::Map<Eigen::Vector4d> q_ws_dot(&dxdt[0]); 
			Eigen::Map<Eigen::Vector3d> v_s_dot(&dxdt[4]);
			Eigen::Map<Eigen::Vector3d> b_g_dot(&dxdt[7]);
			Eigen::Map<Eigen::Vector3d> b_a_dot(&dxdt[10]);
		
			// get interpolated imu measurement at time t
			double t0 = m0_.t_;
			double t1 = m1_.t_;
			double t_span = t1 - t0;
			double c = (t - t0) / t_span;
			Eigen::Vector3d g = (1.0 - c) * m0_.g_ + c * m1_.g_;
			Eigen::Vector3d a = (1.0 - c) * m0_.a_ + c * m1_.a_;

			g = g - b_g; // subtract gyro biases
			a = a - b_a; // subtract accel biases

			// define differential equations
			// ref: Leutenegger et al, 2015
			Eigen::Matrix4d Omega;
			Omega <<    0, -g(2),  g(1), g(0),
							 g(2),     0, -g(0), g(1),
							-g(1),  g(0),     0, g(2),
							-g(0), -g(1), -g(2),    0;
			q_ws_dot = 0.5 * Omega * q_ws;
			
			Eigen::Quaterniond q_ws_quat(q_ws(3),q_ws(0),q_ws(1),q_ws(3)); // eigen uses w,i,j,k
			Eigen::Matrix3d C_sw = q_ws_quat.toRotationMatrix().inverse();
			Eigen::Matrix3d omega_cross;
			omega_cross <<    0, -g(2),  g(1),
										 g(2),     0, -g(0),
    	              -g(1),   g(0),    0;
			v_s_dot = a + (C_sw * g_w_) - (omega_cross * v_s);
	
			b_g_dot = Eigen::Vector3d::Zero();
			b_a_dot = -(1.0 / tau_) * b_a;	
		}

	private:
		ImuMeasurement m0_;
		ImuMeasurement m1_;
		Eigen::Vector3d g_w_;
		double tau_;
};

class ImuVelocityCostFunction : public ceres::CostFunction
{

	typedef std::vector<double> ImuState;

  public:
		ImuVelocityCostFunction(double t0, 
														double t1, 
														std::vector<ImuMeasurement> &imu_measurements,
														ImuParams &params)
			: t0_(t0), t1_(t1), imu_measurements_(imu_measurements), params_(params)
		{
			set_num_residuals(12);
			mutable_parameter_block_sizes()->push_back(4); // orientation at t0 (i,j,k,w)
			mutable_parameter_block_sizes()->push_back(3); // velocity at t0
			mutable_parameter_block_sizes()->push_back(3); // imu gyro biases at t0
			mutable_parameter_block_sizes()->push_back(3); // imu accelerometer biases at t0
			mutable_parameter_block_sizes()->push_back(4); // orientation at t1
			mutable_parameter_block_sizes()->push_back(3); // velocity at t1
			mutable_parameter_block_sizes()->push_back(3); // imu gyro biases at t1
			mutable_parameter_block_sizes()->push_back(3); // imu accelerometer biases at t1
		}

		bool Evaluate(double const* const* parameters,
									double* residuals,
									double** jacobians) const
		{
			// map parameter blocks to eigen containers
			Eigen::Map<const Eigen::Vector4d> q_ws_0(&parameters[0][0]);
			Eigen::Map<const Eigen::Vector3d> v_s_0(&parameters[1][0]);
			Eigen::Map<const Eigen::Vector3d> b_g_0(&parameters[2][0]);
			Eigen::Map<const Eigen::Vector3d> b_a_0(&parameters[3][0]);
			Eigen::Map<const Eigen::Vector4d> q_ws_1(&parameters[4][0]);
			Eigen::Map<const Eigen::Vector3d> v_s_1(&parameters[5][0]);
			Eigen::Map<const Eigen::Vector3d> b_g_1(&parameters[6][0]);
			Eigen::Map<const Eigen::Vector3d> b_a_1(&parameters[7][0]);

			// get rotation matrices
			const Eigen::Quaterniond q_ws_0_quat(q_ws_0(3), q_ws_0(0), q_ws_0(1), q_ws_0(2));
			const Eigen::Quaterniond q_ws_1_quat(q_ws_1(3), q_ws_1(0), q_ws_1(1), q_ws_1(2));
			const Eigen::Matrix3d C_ws_0 = q_ws_0_quat.toRotationMatrix();
			const Eigen::Matrix3d C_sw_0 = C_ws_0.inverse();
			const Eigen::Matrix3d C_ws_1 = q_ws_1_quat.toRotationMatrix();
			const Eigen::Matrix3d C_sw_1 = C_ws_1.inverse();

			// initialize propagated states
			Eigen::Vector4d q_ws_hat(q_ws_0(0), q_ws_0(1), q_ws_0(2), q_ws_0(3));
			Eigen::Vector3d v_s_hat(v_s_0(0), v_s_0(1), v_s_0(2));
			Eigen::Vector3d b_a_hat(b_a_0(0), b_a_0(1), b_a_0(2));

			Eigen::Matrix<double,12,12> F; // jacobian matrix
			Eigen::Matrix<double,12,12> P; // covariance matrix
			Eigen::Matrix<double,12,12> Q; // measurement noise matrix

			F.setIdentity();
			P.setIdentity();
			Q.setIdentity();

			// set up noise matrix
			Q.block<3,3>(0,0) *= params_.sigma_g_;
			Q.block<3,3>(3,3) *= params_.sigma_a_;
			Q.block<3,3>(6,6) *= params_.sigma_b_g_;
			Q.block<3,3>(9,9) *= params_.sigma_b_a_;


			// propagate imu measurements, tracking jacobians and covariance
			for (int i = 1; i < imu_measurements_.size(); i++)
			{
				ImuMeasurement meas_0 = imu_measurements_[i-1];
				ImuMeasurement meas_1 = imu_measurements_[i];

				// if meas_0 is before t0_, interpolate measurement to match t0_
				if (meas_0.t_ < t0_)
				{
					double c = (t0_ - meas_0.t_) / (meas_1.t_ - meas_0.t_);
					meas_0.t_ = t0_;
					meas_0.g_ = (1.0 - c) * meas_0.g_ + c * meas_1.g_;
					meas_0.a_ = (1.0 - c) * meas_0.a_ + c * meas_1.a_;
				}

				// if meas_1 is after t1_, interpolate measurement to match t1_
				if (meas_1.t_ > t1_)
				{
					double c = (t1_ - meas_0.t_) / (meas_1.t_ - meas_0.t_);
					meas_1.t_ = t1_;
					meas_1.g_ = (1.0 - c) * meas_0.g_ + c * meas_1.g_;
					meas_1.a_ = (1.0 - c) * meas_0.a_ + c * meas_1.a_;
				}

				double delta_t = meas_1.t_ - meas_0.t_;

				// get average of imu readings
				Eigen::Vector3d omega_true = (meas_0.g_ + meas_1.g_) / 2.0;
				Eigen::Vector3d acc_true = (meas_0.a_ + meas_1.a_) / 2.0;

				// integrate measurements using Runge-Kutta 4
				std::vector<double> x0;
				for (int j = 0; j < 4; j++) x0.push_back(q_ws_hat(j));
				for (int j = 0; j < 3; j++) x0.push_back(v_s_hat(j));
				for (int j = 0; j < 3; j++) x0.push_back(b_g_0(j));
				for (int j = 0; j < 3; j++) x0.push_back(b_a_hat(j));
				double t_step = delta_t / 10.0;
				ImuIntegrator imu_int(meas_0, meas_1, params_.g_, params_.b_a_tau_);
				boost::numeric::odeint::runge_kutta4<std::vector<double>> stepper;
				boost::numeric::odeint::integrate_const(stepper, imu_int, x0, meas_0.t_, meas_1.t_, t_step);
				q_ws_hat << x0[0], x0[1], x0[2], x0[3];
				v_s_hat << x0[4], x0[5], x0[6];
				b_a_hat << x0[10], x0[11], x0[12];

				// get orientation matrices
				Eigen::Quaterniond q_ws_hat_quat(q_ws_hat(3),q_ws_hat(0),q_ws_hat(1),q_ws_hat(2));
				Eigen::Matrix3d C_ws_hat = q_ws_hat_quat.toRotationMatrix();
				Eigen::Matrix3d C_sw_hat = C_ws_hat.inverse();

				// calculate continuous time jacobian
				Eigen::Matrix<double,12,12> F_c = Eigen::Matrix<double,12,12>::Zero();
				F_c.block<3,3>(0,6) = C_ws_hat;
				Eigen::Matrix3d g_w_cross = Eigen::Matrix3d::Zero();
				g_w_cross(0,1) = -params_.g_;
				g_w_cross(1,0) = params_.g_;
				F_c.block<3,3>(3,0) = -C_sw_hat * g_w_cross;
				Eigen::Matrix3d omega_cross;
				omega_cross << 0, -omega_true(2), omega_true(1),
								omega_true(2), 0, -omega_true(0),
								-omega_true(1), omega_true(0), 0;
				F_c.block<3,3>(3,3) = -omega_cross;
				Eigen::Matrix3d v_s_hat_cross;
				v_s_hat_cross << 0, -v_s_hat(2), v_s_hat(1),
								v_s_hat(2), 0, -v_s_hat(0),
								-v_s_hat(1), v_s_hat(0), 0;
				F_c.block<3,3>(3,6) = -v_s_hat_cross;
				F_c.block<3,3>(3,9) = -1.0 * Eigen::Matrix3d::Identity();
				F_c.block<3,3>(9,9) = (-1.0 / params_.b_a_tau_) * Eigen::Matrix3d::Identity();

				// approximate discrete time jacobian
				Eigen::Matrix<double,12,12> F_d = Eigen::Matrix<double,12,12>::Identity();
				F_d = F_d + F_c * delta_t;

				F = F_d * F; // update total jacobian

				Eigen::Matrix<double,12,12> G = Eigen::Matrix<double,12,12>::Identity();
				G.block<3,3>(0,0) = C_ws_hat;

				// update covariance
				P = F_d * P * F_d.transpose() + G * Q * G.transpose();
			}

			// finish jacobian
			Eigen::Matrix<double,12,12> de_dX = Eigen::Matrix<double,12,12>::Identity();
			Eigen::Quaterniond q_ws_hat_quat(q_ws_hat(3),q_ws_hat(0),q_ws_hat(1),q_ws_hat(2));
			Eigen::Quaterniond q_ws_err = q_ws_hat_quat.conjugate() * q_ws_1_quat;
			Eigen::Matrix3d q_err_cross;
			q_err_cross << q_ws_err.w(), -q_ws_err.z(), q_ws_err.y(),
							q_ws_err.z(), q_ws_err.w(), -q_ws_err.x(),
							-q_ws_err.y(), q_ws_err.x(), q_ws_err.w();
			de_dX.block<3,3>(0,0) = q_err_cross;

			F = de_dX * F;

			if (jacobians != NULL)
			{

			}
			
			return true;
		}

	protected:
		double t0_;
		double t1_;
		std::vector<ImuMeasurement> imu_measurements_;
		ImuParams params_;
};
