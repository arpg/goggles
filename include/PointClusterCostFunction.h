#pragma once
#ifndef POINTCLUSTERCOSTFUNCTION_H_
#define POINTCLUSTERCOSTFUNCTION_H_

#include <Eigen/Core>
#include "DataTypes.h"
#include <ErrorInterface.h>
#include <ceres/ceres.h>

class PointClusterCostFunction : public ceres::CostFunction, public ErrorInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointClusterCostFunction(Eigen::Vector3d &target,
                           double weight);

  ~PointClusterCostFunction();

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
  Eigen::Vector3d target_;
  double weight_;
};

#endif