#ifndef QUATERNIONPARAMETERIZATION_H_
#define QUATERNIONPARAMETERIZATION_H_

#include "ceres/ceres.h"

class QuaternionParameterization : public ceres::LocalParemeterization
{
	public:
		
		bool Plus(const double* x, const double* delta,
							double* x_plus_delta) const
		{
			return plus(x, delta, x_plus_delta);
		}

		bool plus(const double* x, const double* delta,
							double* x_plus_delta)
		{
			Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_(delta);
			Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);

			// apply delta to quaternion
			Eigen::Vector4d dq;
			double halfnorm = 0.5 * delta.norm();
			dq.template head<3>() = sinc(halfnorm) * 0.5 * delta;
			dq[3] = cos(halfnorm);

			q = Eigen::Quaterniond(dq) * q;
			q.normalize();

			x_plus_delta[0] = q.coeffs()[0];
			x_plus_delta[1] = q.coeffs()[1];
			x_plus_delta[2] = q.coeffs()[2];
			x_plus_delta[3] = q.coeffs()[3];

			return true;
		}

		bool Minus(const double* x, const double* x_plus_delta,
							 double* delta) const
		{
			return minus(x, x_plus_delta, delta);
		}

		bool minus(const double* x, const double* x_plus_delta,
							 double* delta)
		{
			
		}

		bool ComputeLiftJacobian(const double* x, double* jacobian) const
		{
			return liftJacobian(x, jacobian);
		}

		bool plusJacobian(const double* x, double* jacobian)
		{

		}

		bool liftJacobian(const double* x, double* jacobian)
		{

		}

		bool ComputeJacobian(const double* x, double* jacobian) const
		{
			return plusJacobian(x, jacobian);
		}

		bool VerifyJacobianNumDiff(const double* x,
															 double* jacobian,
															 double* jacobian_num_diff)
		{

		}

		bool Verify(const double* x_raw, double perturbation_magnitude) const
		{
			const ceres::LocalParameterization* casted = 
				dynamic_cast<const ceres::LocalParameterization*>(this);
			
			if (!casted) return false;
			
			// verify plus and minus functions
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

		int GlobalSize() const { return 4; }

		int LocalSize() const { return 3; }

		double sinc(double x)
		{
			if (fabs(x) < 1e-6) 
			{
				return sin(x) / x;
			}
			else
			{
				static const double c_2 = 1.0 / 6.0;
				static const double c_4 = 1.0 / 120.0;
				static const double c_6 = 1.0 / 5040.0;
				const double x_2 = x * x;
				const double x_4 = x_2 * x_2;
				const double x_6 = x_2 * x_2 * x_2;
				return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
			}
		}
};
