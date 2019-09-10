#include <ceres/ceres.h>

#pragma once
#ifndef QUATERNIONPARAMETERIZATION_H_
#define QUATERNIONPARAMETERIZATION_H_


class QuaternionParameterization : public ceres::LocalParameterization
{
	public:

		Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& dAlpha) const
		{
			Eigen::Vector4d dq;
			double halfnorm = 0.5 * dAlpha.norm();
			dq.head(3) = sinc(halfnorm) * 0.5 * dAlpha;
			dq[3] = cos(halfnorm);
			return Eigen::Quaterniond(dq);
		}
		
		bool Plus(const double* x, const double* delta,
							double* x_plus_delta) const
		{
			return plus(x, delta, x_plus_delta);
		}

		bool plus(const double* x, const double* delta,
							double* x_plus_delta) const
		{
			Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_mapped(delta);
			const Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);

			// apply delta to quaternion
			Eigen::Quaterniond q_plus_delta = DeltaQ(delta_mapped);
			q_plus_delta.normalize();
			
			const Eigen::Vector4d q_vec = q_plus_delta.coeffs();
			x_plus_delta[0] = q_vec[0];
			x_plus_delta[1] = q_vec[1];
			x_plus_delta[2] = q_vec[2];
			x_plus_delta[3] = q_vec[3];

			if (q_plus_delta.norm() - 1.0 < -1.0e-15 || 
						q_plus_delta.norm() - 1.0 > 1.0e-15)
				LOG(ERROR) << "not a unit quaternion";

			return true;
		}

		bool Minus(const double* x, const double* x_plus_delta,
							 double* delta) const
		{
			return minus(x, x_plus_delta, delta);
		}

		bool minus(const double* x, const double* x_plus_delta,
							 double* delta) const
		{
			// put inputs into quaternion form
			const Eigen::Quaterniond q_plus_delta(x_plus_delta[3], x_plus_delta[0],
																						x_plus_delta[1], x_plus_delta[2]);
			const Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);

			// map output 
			Eigen::Map<Eigen::Vector3d> delta_q(delta);

			// find quaternion difference and return twice the imaginary part
			delta_q = 2.0 * (q_plus_delta * q.inverse()).coeffs().template head<3>();
			return true;
		}

		bool ComputeLiftJacobian(const double* x, double* jacobian) const
		{
			return liftJacobian(x, jacobian);
		}

		bool plusJacobian(const double* x, double* jacobian) const
		{
			Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J(jacobian);
			J.setZero();

			Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);
			
			Eigen::Matrix<double,4,3> S = Eigen::Matrix<double,4,3>::Zero();
			S.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 0.5;
			J = qplus(q) * S;

			return true;
		}

		bool liftJacobian(const double* x, double* jacobian) const
		{
			Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> J_lift(jacobian);
			J_lift.setZero();
			const Eigen::Quaterniond q_inv(x[3], -x[0], -x[1], -x[2]);
			Eigen::Matrix4d Qplus = qplus(q_inv);
			Eigen::Matrix<double,3,4> Jq_pinv;
			Jq_pinv.bottomRightCorner<3,1>().setZero();
			Jq_pinv.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() * 2.0;
			J_lift = Jq_pinv * Qplus;

			return true;
		}

		bool ComputeJacobian(const double* x, double* jacobian) const
		{
			return plusJacobian(x, jacobian);
		}

		bool VerifyJacobianNumDiff(const double* x,
															 double* jacobian,
															 double* jacobian_num_diff)
		{
			plusJacobian(x, jacobian);
			Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> Jp(jacobian);
			Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> Jpn(jacobian_num_diff);
			double dx = 1.0e-9;
			Eigen::Vector4d xp;
			Eigen::Vector4d xm;
			for (size_t i = 0; i < 3; i++)
			{
				Eigen::Vector3d delta;
				delta.setZero();
				delta[i] = dx;
				Plus(x, delta.data(), xp.data());
				delta[i] = -dx;
				Plus(x, delta.data(), xm.data());
				Jpn.col(i) = (xp - xm) / (2.0 * dx);
			}
			if ((Jp - Jpn).norm() < 1.0e-6) return true;
			else return false;
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

		double sinc(double x) const
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

		Eigen::Matrix4d oplus(const Eigen::Quaterniond &q_BC) const
		{
			Eigen::Vector4d q = q_BC.coeffs();
			Eigen::Matrix4d Q;

			Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
			Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
			Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
			Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
		
			return Q;
		}

		Eigen::Matrix4d qplus(const Eigen::Quaterniond &q_BC) const
		{
			Eigen::Vector4d q = q_BC.coeffs();
			Eigen::Matrix4d Q;

			Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
			Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
			Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
			Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
			
			return Q;
		}
};

#endif
