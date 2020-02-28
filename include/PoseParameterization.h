#pragma once
#ifndef POSEPARAMETERIZATION_H_
#define POSEPARAMETERIZATION_H_

#include "ceres/ceres.h"
#include <Transformation.h>
#include <Eigen/Core>

class PoseParameterization : public ceres::LocalParameterization
{
public:
  
  PoseParameterization();

  ~PoseParameterization();

  bool Plus(const double* x, const double* delta, double* x_plus_delta) const;

  bool plus(const double* x, const double* delta, double* x_plus_delta) const;

  bool Minus(const double* x, const double* x_plus_delta, double* delta) const;

  bool minus(const double* x, const double* x_plus_delta, double* delta) const;

  bool ComputeLiftJacobian(const double* x, double* jacobian) const;

  bool plusJacobian(const double* x, double* jacobian) const;

  bool liftJacobian(const double* x, double* jacobian) const;

  bool ComputeJacobian(const double* x, double* jacobian) const;

  bool VerifyJacobianNumDiff(const double* x, 
                             double* jacobian, 
                             double* jacobian_num_diff) const;

  bool Verify(const double* x_raw, double perturbation_magnitude) const;

  int GlobalSize() const {return 7;}

  int LocalSize() const {return 6;}

  double sinc(double x) const;

  Eigen::Matrix4d oplus(const Eigen::Quaterniond &q_BC) const;

  Eigen::Matrix4d qplus(const Eigen::Quaterniond &q_BC) const;
};

#endif