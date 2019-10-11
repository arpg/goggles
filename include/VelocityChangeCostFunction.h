#pragma once
#ifndef VELOCITYCHANGECOSTFUNCTION_H_
#define VELOCITYCHANGECOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include <ErrorInterface.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class VelocityChangeCostFunction : public ceres::CostFunction, public ErrorInterface
{
public:

  VelocityChangeCostFunction(double t);

  ~VelocityChangeCostFunction();

  bool Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const;

  size_t ResidualDim() const;

  bool EvaluateWithMinimalJacobians(
    double const* const* parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const;

protected:
  double delta_t_;
};

#endif