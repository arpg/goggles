#include <MarginalizationError.h>

inline void Resize(Eigen::MatrixXd& mat, int rows, int cols)
{
  Eigen::MatrixXd tmp(rows,cols);
  const int common_rows = std::min(rows, (int)mat.rows());
  const int common_cols = std::min(cols, (int)mat.cols());
  tmp.topLeftCorner(common_rows, common_cols) 
    = mat.topLeftCorner(common_rows, common_cols);
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

MarginalizationError::MarginalizationError(std::shared_ptr<Map> map)
{
  map_ptr_ = map;
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
  std::shared_ptr<ErrorInterface> err_interface_ptr = 
    map_ptr_->GetErrorInterfacePtr(residual_block_id);

  if (!err_interface_ptr)
    return false;

  error_computation_valid_ = false;

  // get associated parameter blocks
  Map::ParameterBlockCollection param_blks = map_ptr_->GetParameterBlocks(
    residual_block_id);

  // go through all the associated parameter blocks
  for (size_t i = 0; i < param_blks.size(); i++)
  {
    Map::ParameterBlockInfo parameter_block_spec = param_blks[i];

    // check if parameter block is already connected to the marginalization error
    std::map<uint64_t, size_t>::iterator it = 
      parameter_block_id_2_block_info_idx_.find(parameter_block_spec.first);
    
    // if parameter block is not connected, add it
    if (it == parameter_block_id_2_block_info_idx_.end())
    {
      // is this block a ray delta?
      bool is_delta = false;
      if (std::dynamic_pointer_cast<DeltaParameterBlock>(
        parameter_block_spec.second))
      {
        is_delta = true;
      }

      // resize equation system
      const size_t orig_size = H_.cols();
      size_t additional_size = 0;
      if (!parameter_block_spec.second->IsFixed() && !is_delta)
        additional_size = parameter_block_spec.second->GetMinimalDimension();

      if (additional_size > 0) // will be zero for fixed parameter blocks
      {
        Resize(H_, orig_size + additional_size, orig_size + additional_size);
        Resize(b0_, orig_size + additional_size);

        H_.bottomRightCorner(H_.rows(), additional_size).setZero();
        H_.bottomRightCorner(additional_size, H_.rows()).setZero();
        b0_.tail(additional_size).setZero();
      } 

      // update bookkeeping
      // not adding delta parameter blocks for now
      ParameterBlockInfo info(parameter_block_spec.second,
                              parameter_block_spec.first,
                              orig_size,
                              is_delta);
      param_block_info_.push_back(info);
      parameter_block_id_2_block_info_idx_.insert(
        std::pair<uint64_t, size_t>(parameter_block_spec.first,
                                    param_block_info_.size() - 1.0));

      // update base type bookkeeping
      base_t::mutable_parameter_block_sizes()->push_back(info.dimension);
    }
  }
  
  base_t::set_num_residuals(H_.cols());
  double** parameters_raw = new double*[param_blks.size()];
  Eigen::VectorXd residuals_eigen(err_interface_ptr->ResidualDim());
  double* residuals_raw = residuals_eigen.data();

  double** jacobians_raw = new double*[param_blks.size()];
  std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::aligned_allocator<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> jacobians_eigen(
      param_blks.size());

  double** jacobians_minimal_raw = new double*[param_blks.size()];
  std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::aligned_allocator<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> jacobians_minimal_eigen(
      param_blks.size());

  for (size_t i = 0; i < param_blks.size(); i++)
  {
    size_t idx = parameter_block_id_2_block_info_idx_[param_blks[i].first];
    
    parameters_raw[i] = param_block_info_[idx].linearization_point.get();

    jacobians_eigen[i].resize(err_interface_ptr->ResidualDim(),
                              param_blks[i].second->GetDimension());
    jacobians_raw[i] = jacobians_eigen[i].data();

    jacobians_minimal_eigen[i].resize(err_interface_ptr->ResidualDim(),
                                      param_blks[i].second->GetMinimalDimension());
    jacobians_minimal_raw[i] = jacobians_minimal_eigen[i].data();
  }

  // evaluate the residual
  err_interface_ptr->EvaluateWithMinimalJacobians(parameters_raw, 
                                                  residuals_raw, 
                                                  jacobians_raw,
                                                  jacobians_minimal_raw);
  
  // apply loss function
  const ceres::LossFunction* loss_func = 
    map_ptr_->GetResidualBlockIdToInfoMap().find(
      residual_block_id)->second.loss_function_ptr;

  if (loss_func)
  {
    const double sq_norm = residuals_eigen.transpose() * residuals_eigen;
    double rho[3];
    loss_func->Evaluate(sq_norm, rho);
    const double sqrt_rho1 = sqrt(rho[1]);
    double residual_scaling;
    double alpha_sq_norm;
    if ((sq_norm == 0.0) || (rho[2] <= 0.0))
    {
      residual_scaling = sqrt_rho1;
      alpha_sq_norm = 0.0;
    }
    else
    {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      
      if (std::isnan(alpha))
        LOG(FATAL) << "alpha has nan value";

      residual_scaling = sqrt_rho1 / (1.0 - alpha);
      alpha_sq_norm = alpha / sq_norm;
    }

    // correct jacobians
    for (size_t i = 0; i < param_blks.size(); i++)
    {
      jacobians_minimal_eigen[i] = sqrt_rho1
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
    Map::ParameterBlockInfo parameter_block_spec = param_blks[i];

    ParameterBlockInfo parameter_block_info_i = param_block_info_.at(
      parameter_block_id_2_block_info_idx_[parameter_block_spec.first]);

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
      ParameterBlockInfo parameter_block_info_j = param_block_info_.at(
        parameter_block_id_2_block_info_idx_[param_blks[j].first]);

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
  }
  map_ptr_->RemoveResidualBlock(residual_block_id);

  delete[] parameters_raw;
  delete[] jacobians_raw;
  delete[] jacobians_minimal_raw;

  Check();
  return true;
}

void MarginalizationError::GetParameterBlockPtrs(
  std::vector<std::shared_ptr<ParameterBlock>> &parameter_block_ptrs)
{
  parameter_block_ptrs.clear();
  for (size_t i = 0; i < param_block_info_.size(); i++)
  {
    parameter_block_ptrs.push_back(
      param_block_info_[i].parameter_block_ptr);
  }
}

void MarginalizationError::Check()
{
  // check number of parameter blocks
  if (param_block_info_.size() != base_t::parameter_block_sizes().size())
  {
    LOG(FATAL) << "check failed: number of parameter blocks in local"
               << " bookkeeping (" << param_block_info_.size() 
               << ") does not match number in base type ("
               << base_t::parameter_block_sizes().size() << ")";
  }

  int total_size = 0;
  for (size_t i = 0; i < param_block_info_.size(); i++)
  {
    total_size += param_block_info_[i].minimal_dimension;

    // check parameter block size
    if (param_block_info_[i].dimension != 
      size_t(base_t::parameter_block_sizes()[i]))
    {
      LOG(FATAL) << "check failed: size of parameter block " << i 
                 << " (" << param_block_info_[i].dimension 
                 << ") does not match size in base type ("
                 << parameter_block_sizes()[i] << ")";
    }

    // check parameter block existence
    if (!map_ptr_->ParameterBlockExists(
      param_block_info_[i].parameter_block_id))
    {
      LOG(FATAL) << "check failed: parameter block " << i 
                 << " does not exist in the ceres problem";
    }

    // check for proper bookkeeping
    if (parameter_block_id_2_block_info_idx_[
      param_block_info_[i].parameter_block_id] != i)
    {
      LOG(FATAL) << "check failed: index of parameter block " << i 
                 << " is incorrect in internal bookkeeping";
    }
  }

  // check that parameters are contiguous
  for (size_t i = 1; i < param_block_info_.size(); i++)
  {
    if (param_block_info_[i-1].ordering_idx 
      + param_block_info_[i-1].minimal_dimension
      != param_block_info_[i].ordering_idx)
    {
      LOG(FATAL) << "check failed: parameter block " << i-1 
                 << " is not contiguous with parameter block " << i;
    }
  }

  // check residual dimension
  if (base_t::num_residuals() != total_size)
  {
    LOG(FATAL) << "check failed: residual size (" << base_t::num_residuals()
               << ") does not agree with parameter block sizes (" 
               << total_size << ")";
  }
}

bool MarginalizationError::MarginalizeOut(
  const std::vector<uint64_t> & parameter_block_ids)
{
  if (parameter_block_ids.size() == 0)
  {
    return false;
  }

  // make copy so we can manipulate
  std::vector<uint64_t> parameter_block_ids_copy = parameter_block_ids;

  // decide which blocks need to be marginalized
  std::vector<std::pair<int, int>> marginalization_start_idx_and_length_pairs;
  size_t marginalization_parameters = 0;

  // make sure there are no duplicates
  std::sort(parameter_block_ids_copy.begin(), parameter_block_ids_copy.end());
  for (size_t i = 1; i < parameter_block_ids_copy.size(); i++)
  {
    if (parameter_block_ids_copy[i] == parameter_block_ids_copy[i-1])
    {
      parameter_block_ids_copy.erase(parameter_block_ids_copy.begin() + i);
      i--;
    }
  }

  // find start idx and dimension of param blocks in H matrix
  for (size_t i = 0; i < parameter_block_ids_copy.size(); i++)
  {
    std::map<uint64_t, size_t>::iterator it = 
      parameter_block_id_2_block_info_idx_.find(parameter_block_ids_copy[i]);

    if (it == parameter_block_id_2_block_info_idx_.end())
    {
      LOG(ERROR) << "trying to marginalize unconnected unconnected parameter block";
      return false;
    }

    size_t start_idx = param_block_info_.at(it->second).ordering_idx;
    size_t min_dim = param_block_info_.at(it->second).minimal_dimension;

    marginalization_start_idx_and_length_pairs.push_back(
      std::pair<int,int>(start_idx, min_dim));
    marginalization_parameters += min_dim;
  }

  // ensure marginalization pairs are ordered
  std::sort(marginalization_start_idx_and_length_pairs.begin(),
            marginalization_start_idx_and_length_pairs.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) 
            { return left.first < right.first; } 
            );

  // unify contiguous blocks
  for (size_t i = 1; i < marginalization_start_idx_and_length_pairs.size(); i++)
  {
    if (marginalization_start_idx_and_length_pairs.at(i-1).first
        + marginalization_start_idx_and_length_pairs.at(i-1).second
        == marginalization_start_idx_and_length_pairs.at(i).first)
    {
      marginalization_start_idx_and_length_pairs.at(i-1).second +=
        marginalization_start_idx_and_length_pairs.at(i).second;

      marginalization_start_idx_and_length_pairs.erase(
        marginalization_start_idx_and_length_pairs.begin() + i);

      i--;
    }
  }

  error_computation_valid_ = false;

  // actually marginalize states
  if (marginalization_start_idx_and_length_pairs.size() > 0)
  {
    // preconditioner
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(
      H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();

    // scale H and b
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    Eigen::MatrixXd U(H_.rows() - marginalization_parameters,
                      H_.rows() - marginalization_parameters);
    Eigen::MatrixXd V(marginalization_parameters, marginalization_parameters);
    Eigen::MatrixXd W(H_.rows() - marginalization_parameters,
                      marginalization_parameters);
    Eigen::VectorXd b_a(H_.rows() - marginalization_parameters);
    Eigen::VectorXd b_b(marginalization_parameters);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalization_parameters);
    Eigen::VectorXd p_b(marginalization_parameters);

    SplitVector(marginalization_start_idx_and_length_pairs, p, p_a, p_b);

    // split lhs
    SplitSymmetricMatrix(marginalization_start_idx_and_length_pairs, H_, U, W, V);

    // split rhs
    SplitVector(marginalization_start_idx_and_length_pairs, b0_, b_a, b_b);

    // invert marginalization block
    Eigen::MatrixXd V_inverse_sqrt(V.rows(), V.cols());
    Eigen::MatrixXd V1 = 0.5 * (V + V.transpose());
    PseudoInverseSymmSqrt(V1, V_inverse_sqrt);

    // Schur
    Eigen::MatrixXd M = W * V_inverse_sqrt;
    b0_.resize(b_a.rows());
    b0_ = (b_a - M * V_inverse_sqrt.transpose() * b_b);
    H_.resize(U.rows(), U.cols());
    H_ = (U - M * M.transpose());

    // unscale
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }

  // update internal ceres size info
  base_t::set_num_residuals(base_t::num_residuals() - marginalization_parameters);

  // delete bookkeeping
  for (size_t i = 0; i < parameter_block_ids_copy.size(); i++)
  {
    // get parameter block index
    size_t idx = parameter_block_id_2_block_info_idx_[parameter_block_ids_copy[i]];
    int margSize = param_block_info_.at(idx).minimal_dimension;

    // erase parameter block from info vector
    param_block_info_.erase(param_block_info_.begin() + idx);

    // update subsequent entries in info vector and indices map
    for (size_t j = idx; j < param_block_info_.size(); j++)
    {
      param_block_info_.at(j).ordering_idx -= margSize;
      parameter_block_id_2_block_info_idx_.at(
        param_block_info_.at(j).parameter_block_id) -= 1;
    }

    // erase entry in indices map
    parameter_block_id_2_block_info_idx_.erase(parameter_block_ids_copy[i]);

    // update internal bookkeeping
    base_t::mutable_parameter_block_sizes()->erase(
      mutable_parameter_block_sizes()->begin() + idx);
  }

  // check if any residuals are still connected to these parameter blocks
  for (size_t i = 0; i < parameter_block_ids_copy.size(); i++)
  {
    Map::ResidualBlockCollection residuals = map_ptr_->GetResidualBlocks(
      parameter_block_ids_copy[i]);
    if (residuals.size() != 0)
    {
      LOG(FATAL) << "trying to marginalize out a parameter block that is still"
                 << " connected to error terms.";
    }
  }

  // actually remove the parameter blocks
  for (size_t i = 0; i < parameter_block_ids_copy.size(); i++)
  {
    map_ptr_->RemoveParameterBlock(parameter_block_ids_copy[i]);
  }

  Check();

  return true;
}

void MarginalizationError::UpdateErrorComputation()
{
  if (error_computation_valid_)
    return;

  // update error dimension
  base_t::set_num_residuals(H_.cols());

  // preconditioner
  Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(
    H_.diagonal().cwiseSqrt(),1.0e-3);
  Eigen::VectorXd p_inv = p.cwiseInverse();

  // lhs SVD: H = J^T*J = USV^T
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(
    0.5 * p_inv.asDiagonal() * (H_ + H_.transpose()) * p_inv.asDiagonal());

  static const double epsilon = std::numeric_limits<double>::epsilon();
  double tolerance = epsilon * H_.cols()
    * saes.eigenvalues().array().maxCoeff();
  S_ = Eigen::VectorXd(
    (saes.eigenvalues().array() > tolerance).select(
      saes.eigenvalues().array(), 0));
  S_pinv_ = Eigen::VectorXd(
    (saes.eigenvalues().array() > tolerance).select(
      saes.eigenvalues().array().inverse(), 0));

  S_sqrt_ = S_.cwiseSqrt();
  S_pinv_sqrt_ = S_pinv_.cwiseSqrt();

  // assign jacobians
  J_ = (p.asDiagonal() * saes.eigenvectors() * (S_sqrt_.asDiagonal())).transpose();

  // error e0 := (-pinv(J^T) * b)
  Eigen::MatrixXd J_pinv_T = (S_pinv_sqrt_.asDiagonal())
    * saes.eigenvectors().transpose() * p_inv.asDiagonal();

  e0_ = (-J_pinv_T * b0_);

  H_ = J_.transpose() * J_;
  b0_ = -J_.transpose() * e0_;

  error_computation_valid_ = true;
}

template<typename Derived_A, typename Derived_U, 
         typename Derived_W, typename Derived_V>
void MarginalizationError::SplitSymmetricMatrix(
  const std::vector<std::pair<int,int>>& marginalization_start_idx_and_length_pairs,
  const Eigen::MatrixBase<Derived_A>& A,
  const Eigen::MatrixBase<Derived_U>& U,
  const Eigen::MatrixBase<Derived_W>& W,
  const Eigen::MatrixBase<Derived_V>& V)
{
  // sanity checks
  const int size = A.cols();
  if (size != A.rows()) LOG(FATAL) << "A matrix not square";
  if (V.cols() != V.rows()) LOG(FATAL) << "V matrix not square";
  if (V.cols() != W.cols()) LOG(FATAL) << "V width not equal to W  width";
  if (U.rows() != W.rows()) LOG(FATAL) << "U height not equal to W height";
  if (V.rows() + U.rows() != size) 
    LOG(FATAL) << "supplied matrices do not form exact upper triangular blocks"
               << " of the original matrix";
  if (U.cols() != U.rows()) LOG(FATAL) << "U matrix not square";

  std::vector<std::pair<int,int>> marginalization_start_idx_and_length_pairs2 = 
    marginalization_start_idx_and_length_pairs;
  marginalization_start_idx_and_length_pairs2.push_back(
    std::pair<int,int>(size, 0));

  const size_t length = marginalization_start_idx_and_length_pairs2.size();

  int lastIdx_row = 0;
  int start_a_i = 0;
  int start_b_i = 0;
  for (size_t i = 0; i < length; i++)
  {
    int lastIdx_col = 0;
    int start_a_j = 0;
    int start_b_j = 0;
    int thisIdx_row = marginalization_start_idx_and_length_pairs2[i].first;
    const int size_a_i = thisIdx_row - lastIdx_row;
    const int size_b_i = marginalization_start_idx_and_length_pairs2[i].second;
    for (size_t j = 0; j < length; j++)
    {
      int thisIdx_col = marginalization_start_idx_and_length_pairs2[j].first;
      const int size_a_j = thisIdx_col - lastIdx_col;
      const int size_b_j = marginalization_start_idx_and_length_pairs2[j].second;

      if (size_a_j > 0 && size_a_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_U>&>(U).block(start_a_i, start_a_j,
                                                           size_a_i, size_a_j)
          = A.block(lastIdx_row, lastIdx_col, size_a_i, size_a_j);
      }

      if (size_b_j > 0 && size_a_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_W>&>(W).block(start_a_i, start_b_j,
                                                           size_a_i, size_b_j)
          = A.block(lastIdx_row, thisIdx_col, size_a_i, size_b_j);
      }

      if (size_b_j > 0 && size_b_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_V>&>(V).block(start_b_i, start_b_j,
                                                           size_b_i, size_b_j)
          = A.block(thisIdx_row, thisIdx_col, size_b_i, size_b_j);
      }

      lastIdx_col = thisIdx_col + size_b_j;
      start_a_j += size_a_j;
      start_b_j += size_b_j;
    }
    lastIdx_row = thisIdx_row + size_b_i;
    start_a_i += size_a_i;
    start_b_i += size_b_i;
  }
}

template<typename Derived_b, typename Derived_b_a, typename Derived_b_b>
void MarginalizationError::SplitVector(
  const std::vector<std::pair<int,int>>& marginalization_start_idx_and_length_pairs,
  const Eigen::MatrixBase<Derived_b>& b,
  const Eigen::MatrixBase<Derived_b_a>& b_a,
  const Eigen::MatrixBase<Derived_b_b>& b_b)
{
  const int size = b.rows();

  if (b.cols() != 1) LOG(FATAL) << "supplied vector not Nx1";
  if (b_a.cols() != 1) LOG(FATAL) << "supplied vector not Nx1";
  if (b_b.cols() != 1) LOG(FATAL) << "supplied vector not Nx1";
  if (b_a.rows() + b_b.rows() != size) 
    LOG(FATAL) << "supplied split vector sizes not equal to "
               << "original vector size";

  std::vector<std::pair<int,int>> marginalization_start_idx_and_length_pairs2 = 
    marginalization_start_idx_and_length_pairs;
  marginalization_start_idx_and_length_pairs2.push_back(
    std::pair<int,int>(size,0));

  const size_t length = marginalization_start_idx_and_length_pairs2.size();

  int lastIdx_row = 0;
  int start_a_i = 0;
  int start_b_i = 0;
  for (size_t i = 0; i < length; i++)
  {
    int thisIdx_row = marginalization_start_idx_and_length_pairs2[i].first;
    const int size_b_i = marginalization_start_idx_and_length_pairs2[i].second;
    const int size_a_i = thisIdx_row - lastIdx_row;

    if (size_a_i > 0)
    {
      const_cast<Eigen::MatrixBase<Derived_b_a>&>(b_a).segment(
        start_a_i, size_a_i) = b.segment(lastIdx_row, size_a_i);
    }

    if (size_b_i > 0)
    {
      const_cast<Eigen::MatrixBase<Derived_b_b>&>(b_b).segment(
        start_b_i, size_b_i) = b.segment(thisIdx_row, size_b_i);
    }

    lastIdx_row = thisIdx_row + size_b_i;
    start_a_i += size_a_i;
    start_b_i += size_b_i;
  }
}

template<typename Derived>
bool MarginalizationError::PseudoInverseSymm(
  const Eigen::MatrixBase<Derived>& a,
  const Eigen::MatrixBase<Derived>& result,
  double epsilon,
  int *rank)
{
  if (a.rows() != a.cols()) LOG(FATAL) << "supplied matrix is not square";

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols() 
    * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
    * Eigen::VectorXd(
      (saes.eigenvalues().array() > tolerance).select(
        saes.eigenvalues().array().inverse(), 0)).asDiagonal()
    * (saes.eigenvectors().transpose());

  if (rank)
  {
    *rank = 0;
    for (int i = 0; i < a.rows(); i++)
    {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

template<typename Derived>
bool MarginalizationError::PseudoInverseSymmSqrt(
  const Eigen::MatrixBase<Derived>& a,
  const Eigen::MatrixBase<Derived>& result,
  double epsilon,
  int *rank)
{
  if (a.rows() != a.cols()) LOG(FATAL) << "supplied matrix is not square";

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
    * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
    * Eigen::VectorXd(
      Eigen::VectorXd(
        (saes.eigenvalues().array() > tolerance).select(
          saes.eigenvalues().array().inverse(), 0)).array().sqrt()).asDiagonal();

  if (rank)
  {
    *rank = 0;
    for (int i = 0; i < a.rows(); i++)
    {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }
  return true;
}

bool MarginalizationError::ComputeDeltaChi(
  Eigen::VectorXd& Delta_chi) const
{
  Delta_chi.resize(H_.rows());
  for (size_t i = 0; i < param_block_info_.size(); i++)
  {
    if (!param_block_info_[i].parameter_block_ptr->IsFixed())
    {
      Eigen::VectorXd Delta_chi_i(param_block_info_[i].minimal_dimension);

      param_block_info_[i].parameter_block_ptr->Minus(
        param_block_info_[i].linearization_point.get(),
        param_block_info_[i].parameter_block_ptr->GetParameters(),
        Delta_chi_i.data());
      Delta_chi.segment(param_block_info_[i].ordering_idx,
                        param_block_info_[i].minimal_dimension) = Delta_chi_i;
    }
  }
  return true;
}

bool MarginalizationError::ComputeDeltaChi(
  double const* const * parameters,
  Eigen::VectorXd& Delta_chi) const
{
  Delta_chi.resize(H_.rows());
  for (size_t i = 0; i < param_block_info_.size(); i++)
  {
    if (!param_block_info_[i].parameter_block_ptr->IsFixed())
    {

      Eigen::VectorXd Delta_chi_i(param_block_info_[i].minimal_dimension);

       param_block_info_[i].parameter_block_ptr->Minus(
        param_block_info_[i].linearization_point.get(),
        parameters[i],
        Delta_chi_i.data());

      Delta_chi.segment(param_block_info_[i].ordering_idx,
                        param_block_info_[i].minimal_dimension) = Delta_chi_i;
    }
  }
  return true;
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

  for (size_t i = 0; i < param_block_info_.size(); i++)
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
              param_block_info_[i].minimal_dimension);
          Jmin_i = J_.block(0, param_block_info_[i].ordering_idx, e0_.rows(),
            param_block_info_[i].minimal_dimension);
        }
      }
      if (jacobians[i] != NULL)
      {
        // get minimal jacobian
        Eigen::Matrix<
            double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Jmin_i;
        Jmin_i = J_.block(0, param_block_info_[i].ordering_idx, e0_.rows(),
                          param_block_info_[i].minimal_dimension);

        Eigen::Map<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 
            Eigen::RowMajor>> J_i(jacobians[i], e0_.rows(),
                                  param_block_info_[i].dimension);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_lift(
          param_block_info_[i].parameter_block_ptr->GetMinimalDimension(),
          param_block_info_[i].parameter_block_ptr->GetDimension());

        param_block_info_[i].parameter_block_ptr->LiftJacobian(
          param_block_info_[i].linearization_point.get(), J_lift.data());

        J_i = Jmin_i * J_lift;
      }
    }
  }

  Eigen::Map<Eigen::VectorXd> e(residuals, e0_.rows());
  e = e0_ + J_ * Delta_Chi;
  return true;
}