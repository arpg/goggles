#pragma once
#ifndef VELOCITYCHANGECOSTFUNCTION_H_
#define VELOCITYCHANGECOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class VelocityChangeCostFunction : public ceres::CostFunction
{
public:

  VelocityChangeCostFunction(double t);

  ~VelocityChangeCostFunction();

  bool Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const;

protected:
  double delta_t_;
};

#endif