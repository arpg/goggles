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
  MarginalizationError(std::shared_ptr<ceres::Problem> problem);

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

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool ComputeDeltaChi(Eigen::VectorXd& Delta_chi) const;

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool ComputeDeltaChi(double const* const * parameters,
                       Eigen::VectorXd& Delta_chi) const;

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
  
  std::shared_ptr<ceres::Problem> problem_; ///< pointer to the underlying ceres problem

};

#endif