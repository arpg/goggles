#pragma once

#ifndef AHRS_YAW_COST_FUNCTION_H_
#define AHRS_YAW_COST_FUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ErrorInterface.h>
#include <ceres/ceres.h>

class AHRSYawCostFunction : public ceres::CostFunction, public ErrorInterface
{
public: 
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AHRSYawCostFunction();

  AHRSYawCostFunction(Eigen::Quaterniond &q_WS_meas, bool invert);

  ~AHRSYawCostFunction();

  size_t ResidualDim() const;

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

  bool EvaluateWithMinimalJacobians(double const* const* parameters, 
                                    double* residuals, 
                                    double** jacobians,
                                    double** jacobians_minimal) const;
protected:
  Eigen::Quaterniond q_WS_meas_;
};


#endif