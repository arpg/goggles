#pragma once

#ifndef AHRS_ORIENTATION_COST_FUNCTION_H_
#define AHRS_ORIENTATION_COST_FUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ErrorInterface.h>
#include <ceres/ceres.h>

class AHRSOrientationCostFunction : public ceres::CostFunction, public ErrorInterface
{
public: 
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AHRSOrientationCostFunction(Eigen::Quaterniond &delta_q);

  ~AHRSOrientationCostFunction();

  size_t ResidualDim() const;

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

  bool EvaluateWithMinimalJacobians(double const* const* parameters, 
                                    double* residuals, 
                                    double** jacobians,
                                    double** jacobians_minimal) const;
protected:
  Eigen::Quaterniond delta_q_;
}


#endif