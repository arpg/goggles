#include <MarginalizationError.h>

inline void Resize(Eigen::MatrixXd& mat, int rows, int cols)
{
  Eigen::MatrixXd tmp(rows,cols);
  const int common_rows = std::min(rows, (int)mat.rows());
  const int common_cols = std::min(cols, (int)mat.cols());
  tmp.topLoftCorner(common_rows, common_cols) 
    = mat.topLoftCorner(common_rows, common_cols);
  mat.swap(tmp);
}

inline void Resize(Eigen::VectorXd& vec, int size)
{
  if (vec.rows() == 1)
  {
    Eigen::VectorXd tmp(size);
    const int common_size = std::min((int)vec.cols(), size);
    tmp.head(common_size) = vec.head(common_size);
    vec.swap(tmp);
  }
  else
  {
    Eigen::VectorXd tmp(size);
    const int common_size = std::min((int) vec.rows(), size);
    tmp.head(common_size) = vec.head(common_size);
    vec.swap(tmp);
  }
}

MarginalizationError::MarginalizationError(std::shared_ptr<ceres::Problem> problem)
{
  problem_ = problem;
}

MarginalizationError::~MarginalizationError(){}

// Add set of residuals to this marginalization error
bool MarginalizationError::AddResidualBlocks(
  const std::vector<ceres::ResidualBlockId> &residual_block_ids)
{
  for (size_t i = 0; i < residual_block_ids.size(); i++)
  {
    if (!AddResidualBlock(residual_block_ids[i]))
      return false;
  }
  return true;
}

// Linearize a single residual, add it to the marginalization error
// and remove the associated residual block
bool MarginalizationError::AddResidualBlock(
  ceres::ResidualBlockId residual_block_id)
{
  // verify residual block exists
  ceres::CostFunction* cost_func 
    = problem->GetCostFunctionForResidualBlock(residual_block_id);
  if (!cost_func)
    return false;

  ErrorInterface* err_interface_ptr = cost_func;

  // get associated parameter blocks
  // should just be one parameter block unless it's an IMU error
  std::vector<double*> param_blks;
  problem_->GetParameterBlocksForResidualBlock(residual_block_id, &param_blks);

  // go through all the associated parameter blocks
  for (size_t i = 0; i < param_blks.size(); i++)
  {
    // check if parameter block is already connected to the marginalization error
    ParameterBlockInfo info;
    std::map<double*, size_t>::iterator it = 
      parameter_block_id_2_block_info_idx_.find(param_blks[i]);
    
    // if parameter block is not connected, add it
    if (it == parameter_block_id_2_block_info_idx_.end())
    {
      // resize equation system
      const size_t orig_size = H_.cols();
      size_t additional_size = info.minimal_dimension;

      if (additional_size > 0) // will be zero for fixed parameter blocks
      {
        Resize(H_, orig_size + additional_size, orig_size + additional_size);
        Resize(b0_, orig_size + additional_size);

        H_.bottomRightCorner(H_.rows(), additional_size).setZero();
        H_.bottomRightCorner(additional_size, H_.cols()).setZero();
        b0_.tail(additional_size).setZero();
      } 

      // update bookkeeping
      info = ParameterBlockInfo(std::shared_ptr<double>(param_blks[i]),
                                problem_,
                                orig_size);
      parameter_block_info_.push_back(info);
      parameter_block_id_2_block_info_idx_.insert(
        std::pair<double*, size_t>(param_blks[i],
                                   param_block_info_.size() - 1.0))

      // update base type bookkeeping
      base_t::mutable_parameter_block_sizes()->push_back(info.dimension);
    }
    else
    {
      info = parameter_block_info_.at(it->second);
    }
  }

  base_t::set_num_residuals(H_.cols());

  double** parameters_raw = new double*[param_blks.size()];
  Eigen::VectorXd residuals_eigen(err_interface_ptr->ResidualDim());
  double* residuals_raw = residuals_eigen.data();

  double** jacobians_raw = new double*[param_blks.size()];
  std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::alligned_allocator<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> jacobians_eigen(
      param_blks.size());

  double** jacobians_minimal_raw = new double*[param_blks.size()];
  std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::alligned_allocator<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> jacobians_minimal_eigen(
      param_blks.size());

  for (size_t i = 0; i < param_blks.size(); i++)
  {
    size_t idx = parameter_block_id_2_block_info_idx_.find(param_blks[i])->second;
    
    parameters_raw[i] = parameter_block_info_[idx].linearization_point.get();

    jacobians_eigen[i].resize(err_interface_ptr->ResidualDim(),
                              parameter_block_info_[idx].dimension);
    jacobians_raw[i] = jacobians_eigen[i].data();

    jacobians_minimal_eigen[i].resize(err_interface_ptr->ResidualDim(),
                                      parameter_block_info_[idx].minimal_dimension);
    jacobians_minimal_raw[i] = jacobians_minimal_eigen[i].data();
  }

  // evaluate the residual
  // won't work as-is; need to implement error interface
  err_interface_ptr->EvaluateWithMinimalJacobians(parameters_raw, 
                                                  residuals_raw, 
                                                  jacobians_raw,
                                                  jacobians_minimal_raw);

  // apply loss function
  ceres::LossFunction* loss_func = 
    problem_->GetLossFunctionForResidualBlock(residual_block_id);

  if (loss_func)
  {
    const double sq_norm = residuals_eigen.transpose() * residuals_eigen;
    double rho[3];
    loss_func->Evaluate(sq_norm, rho);
    const double sqrt_rho = sqrt(rho);
    double residual_scaling;
    double alpha_sq_norm;
    if ((sq_norm == 0.0) || (rho[2] <= 0.0))
    {
      residual_scaling = sqrt_rho;
      alpha_sq_norm = 0.0;
    }
    else
    {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      if (std::isnan(alpha))
        LOG(FATAL) << "alpha has nan value";

      residual_scaling = sqrt_rho / (1.0 - alpha);
      alpha_sq_norm = alpha / sq_norm;
    }

    // correct jacobians
    // this won't work as-is, need a way to access the minimal jacobian representation
    for (size_t i = 0; i < param_blks.size(); i++)
    {
      jacobians_minimal_eigen[i] = sqrt_rho
        * (jacobians_minimal_eigen[i]
          - alpha_sq_norm * residuals_eigen
          * (residuals_eigen.transpose() * jacobians_minimal_eigen[i]));
    }

    // correct residuals
    residuals_eigen *= residual_scaling;
  }

  // actually add blocks to lhs and rhs
  for (size_t i = 0; i < param_blks.size(); i++)
  {
    ParameterBlockInfo parameter_block_info_i = parameter_block_info_.at(
      parameter_block_id_2_block_info_idx_[param_blks[i]]);

    if (parameter_block_info_i.minimal_dimension == 0)
      continue;

    if (!H_.allFinite())
      LOG(FATAL) << "H matrix has inf values prior to update";

    H_.block(parameter_block_info_i.ordering_idx, 
             parameter_block_info_i.ordering_idx,
             parameter_block_info_i.minimal_dimension,
             parameter_block_info_i.minimal_dimension) += jacobians_minimal_eigen.at(
                i).transpose().eval() * jacobians_minimal_eigen.at(i);
    b0_.segment(parameter_block_info_i.ordering_idx,
                parameter_block_info_i.minimal_dimension) -= jacobians_minimal_eigen.at(
                i).transpose().eval() * residuals_eigen;

    if (!H_.allFinite())
      LOG(FATAL) << "H matrix has inf values after update";

    for (size_t j = 0; j < i; j++)
    {
      ParameterBlockInfo parameter_block_info_j = parameter_block_info_.at(
        parameter_block_id_2_block_info_idx_[param_blks[j]]);

      if (parameter_block_info_j.minimal_dimension == 0)
        continue;

      H_.block(parameter_block_info_i.ordering_idx,
               parameter_block_info_j.ordering_idx,
               parameter_block_info_i.minimal_dimension,
               parameter_block_info_j.minimal_dimension) += 
        jacobians_minimal_eigen.at(i).transpose().eval() 
          * jacobians_minimal_eigen.at(j);

      H_.block(parameter_block_info_j.ordering_idx,
               parameter_block_info_i.ordering_idx,
               parameter_block_info_j.minimal_dimension,
               parameter_block_info_i.minimal_dimension) += 
        jacobians_minimal_eigen.at(j).transpose().eval() 
          * jacobians_minimal_eigen.at(i);
    }

    problem_->RemoveResidualBlock(residual_block_id);

    delete[] parameters_raw;
    delete[] jacobians_raw;
    delete[] jacobians_minimal_raw;
  }

  return true;
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
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool MarginalizationError::EvaluateWithMinimalJacobians(double const* const* parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobians_minimal) const
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
      if (jacobians_minimal != NULL)
      {
        if (jacobians_minimal[i] != NULL)
        {
          Eigen::Map<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor>> Jmin_i(
              jacobians_minimal[i], e0_.rows(),
              parameter_block_info_[i].minimal_dimension);
          Jmin_i = J_.block(0, parameter_block_info_[i].ordering_idx, e0_.rows(),
            parameter_block_info_[i].minimal_dimension);
        }
      }
      if (jacobians[i] != NULL)
      {
        // get minimal jacobian
        Eigen::Matrix<
            double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Jmin_i;
        Jmin_i = J_.block(0, parameter_block_info_[i].ordering_idx, e0_.rows(),
                          parameter_block_info_[i].minimal_dimension);

        Eigen::Map<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 
            Eigen::RowMajor>> J_i(jacobians[i], e0_.rows(),
                                  parameter_block_info_[i].dimension);

        // if current paremeter block represents a quaternion,
        // get overparameterized jacobion
        if (parameter_block_info_[i].dimension 
              != parameter_block_info_[i].minimal_dimension)
        {
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_lift(
              parameter_block_info_[i].minimal_dimension,
              parameter_block_info_[i].dimension);
          QuaternionParameterization qp; 
          qp.liftJacobian(
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