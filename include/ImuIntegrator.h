#pragma once
#ifndef IMUINTEGRATOR_H_
#define IMUINTEGRATOR_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class ImuIntegrator
{
public:
  ImuIntegrator(ImuMeasurement &m0, ImuMeasurement &m1, double g_w, double tau);

  ~ImuIntegrator();
    
  void operator() (const std::vector<double> &x, 
                   std::vector<double> &dxdt, 
                   const double t);

private:
  ImuMeasurement m0_;
  ImuMeasurement m1_;
  Eigen::Vector3d g_w_;
  double tau_;
};

#endif