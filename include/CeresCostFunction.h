#include <boost/numeric/odeint.hpp>

struct ImuMeasurement
{
	Eigen::Vector3d g_; // gyro reading
	Eigen::Vector3d a_; // accelerometer reading
	double t_;          // timestamp
}

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
		PropagateImuIntegrator(ImuMeasurement &m0, ImuMeasurement &m1, double g_w, double tau)
			: m0_(m0), m1_(m1), tau_(tau)
		{
			g_w_ << 0,0,g_w; // define world gravity as in positive Z direction
		}
		
		void operator(const std::vector<double> &x, 
									std::vector<double> &dxdt, 
									const double t)
		{
			// map in and out states to Eigen datatypes for convenience
			Eigen::Map<Eigen::Vector4d> q_ws(x.data()[0]); // sensor-to-world rotation
																										 // assumes i,j,k,w representation
			Eigen::Map<Eigen::Vector3d> v_s(x.data()[4]); // sensor frame velocity
			Eigen::Map<Eigen::Vector3d> b_g(x.data()[7]); // gyro biases
			Eigen::Map<Eigen::Vector3d> b_a(x.data()[10]); // accelerometer biases
			Eigen::Map<Eigen::Vector4d> q_ws_dot(dxdt.data()[0]); 
			Eigen::Map<Eigen::Vector3d> v_s_dot(dxdt.data()[4];
			Eigen::Map<Eigen::Vector3d> b_g_dot(dxdt.data()[7]);
			Eigen::Map<Eigen::Vector3d> b_a_dot(dxdt.data()[10]);
		
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
	
			b_g_dot = 0;
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
														std::vector<ImuMeasurement> &imu_measurements)
			: t0_(t0), t1_(t1), imu_measurements_(imu_measurements)
		{
			set_num_residuals(3);
			mutable_parameter_block_sizes()->push_back(3); // orientation at t0
			mutable_parameter_block_sizes()->push_back(3); // velocity at t0
			mutable_parameter_block_sizes()->push_back(3); // orientation at t1
			mutable_parameter_block_sizes()->push_back(3); // velocity at t1
			mutable_parameter_block_sizes()->push_back(3); // imu gyro biases
			mutabel_parameter_block_sizes()->push_back(3); // imu accelerometer biases
		}

		bool Evaluate(double const* const* parameters,
									double* residuals,
									double** jacobians) const
		{
			return true;
		}

	protected:
		double t0_;
		double t1_;
		std::vector<ImuMeasurement> imu_measurements_;
		
};
