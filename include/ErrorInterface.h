#pragma once
#ifndef ERRORINTERFACE_H_
#define ERRORINTERFACE_H_

#include <Eigen/Core>

class ErrorInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ErrorInterface(){}

  virtual size_t ResidualDim() const = 0;

  virtual bool EvaluateWithMinimalJacobians(
    double const* const* parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const = 0;
};

#endif