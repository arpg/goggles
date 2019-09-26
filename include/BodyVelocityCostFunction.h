#pragma once
#ifndef BODYVELOCITYCOSTFUNCTION_H_
#define BODYVELOCITYCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class BodyVelocityCostFunction : public ceres::CostFunction
{
  public:
    BodyVelocityCostFunction(double doppler,
                              Eigen::Vector3d & target,
                              double weight);

    ~BodyVelocityCostFunction();

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

  protected:
    double doppler_;
    double weight_;
    Eigen::Vector3d target_ray_;
};

#endif