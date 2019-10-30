#pragma once
#ifndef PARAMETERBLOCK_H_
#define PARAMETERBLOCK_H_

#include <glog/logging.h>
#include <stdio.h>
#include <iostream>
#include "ceres/ceres.h"

/// @brief Base class for parameter blocks
class ParameterBlock
{
public:

  /// @brief Default constructor
  ParameterBlock() : id_(0), fixed_(false), local_parameterization_ptr_(0) {}

  /// @brief Destructor
  virtual ~ParameterBlock() {}

  /// @brief Set parameter block ID
  /// @param[in] id A unique ID
  void SetId(uint64_t id)
  {
    id_ = id;
  }

  /// @brief Directly set values of this parameter block
  /// @param[in] parameters Pointer to parameters to be copied into this
  virtual void SetParameters(const double* parameters) = 0;

  /// @brief Set whether this block should be optimized
  /// @param[in] fixed True if the block should not be optimized
  void SetFixed(bool fixed)
  {
    fixed_ = fixed;
  }

  /// @brief Get parameter values
  virtual double* GetParameters() = 0;

  /// @brief Get parameter values
  virtual const double* GetParameters() const = 0;

  /// @brief Get the parameter block ID
  uint64_t GetId() const 
  {
    return id_;
  }

  /// @brief Get the dimension of the parameter block
  virtual size_t GetDimension() const = 0;

  /// @brief Get the dimension of the local parameterization
  virtual size_t GetMinimalDimension() const = 0;

  /// @brief Get whether or not this block is fixed
  bool IsFixed() const
  {
    return fixed_;
  }

  /** \brief Generalization of the addition operation
    * @param[in] x0 the variable 
    * @param[in] delta the perturbation
    * @param[out] x0_plus_delta the perturbed variable
    */
  virtual void Plus(const double* x0, const double* delta, 
    double* x0_plus_delta) const = 0;

  /** \brief The jacobian of the plus operation
    * @param[in] x0 the variable
    * @param[out] jacobian The jacobian
    */
  virtual void PlusJacobian(const double* x0, double* jacobian) const = 0;

  /** \brief Computes the minimal difference between a variable and a perturbed value
    * @param[in] x0 The variable
    * @param[in] x0_plus_delta The perturbed variable
    * @param[out] delta The minimal difference
    */
  virtual void Minus(const double* x0, const double* x0_plus_delta, 
    double* delta) const = 0;

  /** \brief Computes the jacobian to translate from minimal to overparameterized representation
    * @param[in] x0 The variable
    * @param[out] jacobian The lift jacobian
    */
  virtual void LiftJacobian(const double* x0, double* jacobian) const = 0;

  /** \brief Sets the local parameterization
    * @param[in] local_parameterization_ptr Pointer to the local parameterization
    */
  virtual void SetLocalParameterization(
    const ceres::LocalParameterization* local_parameterization_ptr)
  {
    local_parameterization_ptr_ = local_parameterization_ptr;
  }

  /// @brief Retur parameter block type as a string
  virtual std::string GetTypeInfo() const = 0;

protected:

  /// @brief ID of the parameter block
  uint64_t id_;

  /// @brief Whether the parameter block is set constant
  bool fixed_;

  /// @brief The local parameterization object to use
  const ceres::LocalParameterization* local_parameterization_ptr_;
};

#endif