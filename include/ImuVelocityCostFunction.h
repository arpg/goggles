#pragma once
#ifndef IMUVELOCITYCOSTFUNCTION_H_
#define IMUVELOCITYCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class ImuVelocityCostFunction : public ceres::CostFunction
{

  typedef std::vector<double> ImuState;

public:
  ImuVelocityCostFunction(double t0, 
                          double t1, 
                          std::vector<ImuMeasurement> &imu_measurements,
                          ImuParams &params);

  ~ImuVelocityCostFunction();

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

protected:
  double t0_;
  double t1_;
  std::vector<ImuMeasurement> imu_measurements_;
  ImuParams params_;
};

#endif