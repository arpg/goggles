#pragma once
#ifndef VELOCITYMEASCOSTFUNCTION_H_
#define VELOCITYMEASCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>


class VelocityMeasCostFunction : public ceres::CostFunction
{
public:
  VelocityMeasCostFunction(Eigen::Vector3d &v_meas);

  ~VelocityMeasCostFunction();

  bool Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const;

  bool EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const;

  size_t ResidualDim() const;

protected:
  const Eigen::Vector3d v_meas_;
};

#endif

  