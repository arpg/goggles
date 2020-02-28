#pragma once
#ifndef HOMOGENEOUSPOINTPARAMETERIZATION_H_
#define HOMOGENEOUSPOINTPARAMETERIZATION_H_

#include <Eigen/Core>
#include "ceres/ceres.h"

class HomogeneousPointParameterization : public ceres::LocalParameterization
{
public:
  HomogeneousPointParameterization();

  ~HomogeneousPointParameterization();

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
};

#endif