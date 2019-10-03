#include <MarginalizationError.h>

MarginalizationError::MarginalizationError(std::shared_ptr<ceres::Problem> problem)
{
  problem_ = problem;
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
  if (!error_computation_valid_) 
    LOG(FATAL) << "trying to evaluate with invalid error computation";

  Eigen::VectorXd Delta_Chi;
  ComputeDeltaChi(parameters, Delta_Chi);

  // will only work with radar-inertial
  // want to be able to work with radar-only as well
  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {

    }
    if (jacobians[1] != NULL)
    {

    }
    if (jacobians[2] != NULL)
    {

    }
    if (jacobians[3] != NULL)
    {

    }
  }

  Eigen::Map<Eigen::VectorXd> e(residuals, e0_.rows());
  e = e0_ + J_ * Delta_Chi;
}