#include <MarginalizationError.h>

MarginalizationError::MarginalizationError()
{

}

MarginalizationError::~MarginalizationError(){}

bool MarginalizationError::AddResidualBlocks(
  const std::vector<ceres::ResidualBlockId> &residual_block_ids)
{

}

bool MarginalizationError::AddResidualBlock(
  ceres::ResidualBlockId residual_block_id)
{

}

bool MarginalizationError::MarginalizeOut(
  const std::vector<uint64_t> & parameter_block_ids)
{

}

bool MarginalizationError::ComputeDeltaChi(
  Eigen::VectorXd& Delta_chi) const
{

}

bool MarginalizationError::ComputeDeltaChi(
  double const* const * parameters,
  Eigen::VectorXd& Delta_chi) const
{
  
}

bool MarginalizationError::Evaluate(
  double const* const* parameters,
  double* residuals,
  double** jacobians) const
{

}