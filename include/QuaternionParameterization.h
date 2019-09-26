#pragma once
#ifndef QUATERNIONPARAMETERIZATION_H_
#define QUATERNIONPARAMETERIZATION_H_

#include "ceres/ceres.h"

class QuaternionParameterization : public ceres::LocalParameterization
{
  public:

    QuaternionParameterization();

    ~QuaternionParameterization();

    Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& dAlpha) const;
    
    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const;

    bool plus(const double* x, const double* delta,
              double* x_plus_delta) const;

    bool Minus(const double* x, const double* x_plus_delta,
               double* delta) const;

    bool minus(const double* x, const double* x_plus_delta,
               double* delta) const;

    bool ComputeLiftJacobian(const double* x, double* jacobian) const;

    bool plusJacobian(const double* x, double* jacobian) const;

    bool liftJacobian(const double* x, double* jacobian) const;

    bool ComputeJacobian(const double* x, double* jacobian) const;

    bool VerifyJacobianNumDiff(const double* x,
                               double* jacobian,
                               double* jacobian_num_diff);

    bool Verify(const double* x_raw, double perturbation_magnitude) const;

    int GlobalSize() const { return 4; }

    int LocalSize() const { return 3; }

    double sinc(double x) const;

    Eigen::Matrix4d oplus(const Eigen::Quaterniond &q_BC) const;

    Eigen::Matrix4d qplus(const Eigen::Quaterniond &q_BC) const;
};

#endif
