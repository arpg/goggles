#pragma once
#ifndef MARGINALIZATIONERROR_H_
#define MARGINALIZATIONERROR_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class MarginalizationError : public ceres::CostFunction
{
public:
  MarginalizationError();

  ~MarginalizationError();

  bool AddResidualBlocks(
    const std::vector<ceres::ResidualBlockId> &residual_block_ids);

  bool AddResidualBlock(
    ceres::ResidualBlockId residual_block_id);

  bool MarginalizeOut(const std::vector<uint64_t> & parameter_block_ids);

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

protected:

};

#endif