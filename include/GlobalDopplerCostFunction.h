#pragma once
#ifndef GLOBALDOPPLERCOSTFUNCTION_H_
#define GLOBALDOPPLERCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ErrorInterface.h>
#include <ceres/ceres.h>

class GlobalDopplerCostFunction : public ceres::CostFunction, public ErrorInterface
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GlobalDopplerCostFunction(double doppler,
                              Eigen::Vector3d &target,
                              Eigen::Matrix3d &radar_to_imu_mat,
                              double weight,
                              double d);

    ~GlobalDopplerCostFunction();

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
    double doppler_;
    double weight_;
    double d_;
    Eigen::Vector3d target_ray_;
};

#endif