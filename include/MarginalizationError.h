#pragma once
#ifndef MARGINALIZATIONERROR_H_
#define MARGINALIZATIONERROR_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include <ErrorInterface.h>
#include <Map.h>
#include <VelocityParameterBlock.h>
#include <DeltaParameterBlock.h>
#include <OrientationParameterBlock.h>
#include <BiasParameterBlock.h>
#include "DataTypes.h"
#include "ceres/ceres.h"
#include <algorithm>

class MarginalizationError : public ceres::CostFunction, public ErrorInterface
{
public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef ceres::CostFunction base_t;

  MarginalizationError(std::shared_ptr<Map> map);

  ~MarginalizationError();

  bool AddResidualBlocks(
    const std::vector<ceres::ResidualBlockId> &residual_block_ids);

  bool AddResidualBlock(
    ceres::ResidualBlockId residual_block_id);

  void GetParameterBlockPtrs(std::vector<std::shared_ptr<ParameterBlock>> 
    &parameter_block_ptrs);

  bool MarginalizeOut(const std::vector<uint64_t>& parameter_block_ids);

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

  bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobians_minimal) const;

  size_t ResidualDim() const
  {
    return base_t::num_residuals();
  }

  void UpdateErrorComputation();

protected:

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool ComputeDeltaChi(Eigen::VectorXd& Delta_chi) const;

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool ComputeDeltaChi(double const* const * parameters,
                       Eigen::VectorXd& Delta_chi) const;

  /// \brief checks the internal data structure
  void Check();

  /// \brief Split matrix for Schur complement operation
  template<typename Derived_A, typename Derived_U, typename Derived_W,
    typename Derived_V>
  static void SplitSymmetricMatrix(
    const std::vector<std::pair<int,int>>& marginalization_start_idx_and_length_pairs,
    const Eigen::MatrixBase<Derived_A>& A,
    const Eigen::MatrixBase<Derived_U>& U,
    const Eigen::MatrixBase<Derived_W>& W,
    const Eigen::MatrixBase<Derived_V>& V);

  /// \brief Split vector for Schur complement operation
  template<typename Derived_b, typename Derived_b_a, typename Derived_b_b>
  static void SplitVector(
    const std::vector<std::pair<int,int>>& marginalization_start_idx_and_length_pairs,
    const Eigen::MatrixBase<Derived_b>& b,
    const Eigen::MatrixBase<Derived_b_a>& b_a,
    const Eigen::MatrixBase<Derived_b_b>& b_b);

  /// \brief Pseudo inversion of a symmetric matrix
  template<typename Derived>
  static bool PseudoInverseSymm(
    const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& result,
    double epsilon = std::numeric_limits<typename Derived::Scalar>::epsilon(),
    int *rank = 0);

  template<typename Derived>
  bool PseudoInverseSymmSqrt(
    const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& result,
    double epsilon = std::numeric_limits<typename Derived::Scalar>::epsilon(),
    int *rank = NULL);

  /// @name The internal storage of the linearised system.
  /// lhs and rhs:
  /// H*delta_Chi = _b - H*Delta_Chi .
  /// the lhs Hessian matrix is decomposed as H = J^T*J = U*S*U^T ,
  /// the rhs is decomposed as b - H*Delta_Chi = -J^T * (-pinv(J^T) * _b + J*Delta_Chi) ,
  /// i.e. we have the ceres standard form with weighted Jacobians J,
  /// an identity information matrix, and an error
  /// e = -pinv(J^T) * b + J*Delta_Chi .
  /// e = e0 + J*Delta_Chi 
  /// @{
  Eigen::MatrixXd H_;  ///< lhs - Hessian
  Eigen::VectorXd b0_;  ///<  rhs constant part
  Eigen::VectorXd e0_;  ///<  _e0 := pinv(J^T) * _b0
  Eigen::MatrixXd J_;  ///<  Jacobian such that _J^T * J == _H
  Eigen::MatrixXd U_;  ///<  H_ = _U*_S*_U^T lhs Eigen decomposition
  Eigen::VectorXd S_;  ///<  singular values
  Eigen::VectorXd S_sqrt_;  ///<  cwise sqrt of _S, i.e. _S_sqrt*_S_sqrt=_S; _J=_U^T*_S_sqrt
  Eigen::VectorXd S_pinv_;  ///<  pseudo inverse of _S
  Eigen::VectorXd S_pinv_sqrt_;  ///<  cwise sqrt of _S_pinv, i.e. pinv(J^T)=_U^T*_S_pinv_sqrt
  Eigen::VectorXd p_;
  Eigen::VectorXd p_inv_;
  volatile bool error_computation_valid_;  ///<  adding residual blocks will invalidate this. before optimizing, call updateErrorComputation()
  
  std::shared_ptr<Map> map_ptr_; ///< pointer to the underlying ceres problem

  struct ParameterBlockInfo
  {
    std::shared_ptr<ParameterBlock> parameter_block_ptr;
    uint64_t parameter_block_id;
    size_t ordering_idx;
    size_t dimension;
    size_t minimal_dimension;
    std::shared_ptr<double> linearization_point;
    bool is_delta;

    ParameterBlockInfo()
      : parameter_block_ptr(std::shared_ptr<ParameterBlock>()),
        parameter_block_id(0),
        ordering_idx(0),
        dimension(0),
        minimal_dimension(0)
    { 
    }
    /*
    ~ParameterBlockInfo()
    {
      LOG(ERROR) << "resetting lin point";
      linearization_point.reset();
      LOG(ERROR) << "resetting param block";
      parameter_block_ptr.reset();
      LOG(ERROR) << "info deleted";
    }
    */

    ParameterBlockInfo(std::shared_ptr<ParameterBlock> param_block_ptr,
                       uint64_t param_block_id,
                       size_t idx,
                       bool delta)
    : parameter_block_ptr(param_block_ptr),
        parameter_block_id(param_block_id),
        ordering_idx(idx),
        is_delta(delta)
    {
      parameter_block_ptr = param_block_ptr;
      parameter_block_id = param_block_id;
      ordering_idx = idx;
      is_delta = delta;

      dimension = parameter_block_ptr->GetDimension();
      minimal_dimension = parameter_block_ptr->GetMinimalDimension();

      if (parameter_block_ptr->IsFixed() || is_delta)
      {
        minimal_dimension = 0;
      }

      linearization_point.reset(new double[dimension],
                               std::default_delete<double[]>());
      ResetLinearizationPoint(parameter_block_ptr);
    }

    void ResetLinearizationPoint(std::shared_ptr<ParameterBlock> param_ptr)
    {
      memcpy(linearization_point.get(), param_ptr->GetParameters(), 
        dimension * sizeof(double));
    }
  };

  std::vector<ParameterBlockInfo> param_block_info_;
  std::map<uint64_t, size_t> parameter_block_id_2_block_info_idx_;
};

#endif