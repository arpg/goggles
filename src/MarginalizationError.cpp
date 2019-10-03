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
  for (size_t i = 0; i < parameter_block_info_.size(); i++)
  {
    if (jacobians != NULL)
    {
      if (jacobians[i] != NULL)
      {
        // get minimal jacobian
        Eigen::Matrix<
            double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Jmin_i;
        Jmin_i = J_.block(0, paremeter_block_info_[i].ordering_idx, e0_.rows(),
                          parameter_block_info_[i].minimal_dimension);

        Eigen::Map<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 
            Eigen::RowMajor>> J_i(jacobians[i], e0_.rows(),
                                  parameter_block_info_[i].dimension);

        // if current paremeter block is overparameterized,
        // get overparameterized jacobion
        if (parameter_block_info_[i].dimension 
              != parameter_block_info_.minimal_dimension)
        {
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_lift(
              parameter_block_info_[i].minimal_dimension,
              parameter_block_info_[i].dimension);
          parameter_block_info_[i].parameter_block_ptr->
              local_parameterization()->liftJacobian(
                  parameter_block_info_[i].linearization_point.get(), J_lift.data());
          J_i = Jmin_i * J_lift;
        }
        else
        {
          J_i = Jmin_i;
        }
      }
    }
  }

  Eigen::Map<Eigen::VectorXd> e(residuals, e0_.rows());
  e = e0_ + J_ * Delta_Chi;

  return true;
}