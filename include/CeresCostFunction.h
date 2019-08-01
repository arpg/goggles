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

		void PropagateImuState(const ImuState &x, ImuState &dxdt, const double t)
		{
			// map in and out states to Eigen datatypes for convenience
			Eigen::Map<Eigen::Quaterniond> q(x.data()[0]);
			Eigen::Map<Eigen::Vector3d> v(x.data()[4]);
			Eigen::Map<Eigen::Vector3d> b_g(x.data()[7]);
			Eigen::Map<Eigen::Vector3d> b_a(x.data()[10]);
			Eigen::Map<Eigen::Quaterniond> q_dot(dxdt.data()[0]);
			Eigen::Map<Eigen::Vector3d> v_dot(dxdt.data()[4];
			Eigen::Map<Eigen::Vector3d> b_g(dxdt.data()[7]);
			Eigen::Map<Eigen::Vector3d> b_a(dxdt.data()[10]);

			// define differential equations

		}
};
