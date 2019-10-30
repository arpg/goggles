#pragma once
#ifndef GLOBALVELOCITYMEASCOSTFUNCTION_H_
#define GLOBALVELOCITYMEASCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include <ErrorInterface.h>
#include "DataTypes.h"
#include <ceres/ceres.h>


class GlobalVelocityMeasCostFunction : public ceres::CostFunction, public ErrorInterface
{
public:
  GlobalVelocityMeasCostFunction(Eigen::Vector3d &vicon_v,
                                 Eigen::Vector3d &radar_v);

  ~GlobalVelocityMeasCostFunction();

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
  const Eigen::Vector3d radar_v_;
  const Eigen::Vector3d vicon_v_;
};

#endif

  