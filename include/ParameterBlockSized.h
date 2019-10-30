#pragma once
#ifndef PARAMETERBLOCKSIZED_H_
#define PARAMETERBLOCKSIZED_H_

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <ParameterBlock.h>
#include <Eigen/Core>


/** @brief base class for parameter blocks
  * @tparam Dim full dimension of the parameter block
  * @tparam MinDim minimal dimension of the parameter block
  * @tparam T the type of parameter block
  */
template<int Dim, int MinDim, class T>
class ParameterBlockSized : public ParameterBlock
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief full dimension of the parameter block
  static const int dimension = Dim;

  /// @brief minimal dimension of the parameter block
  static const int minimal_dimension = MinDim;

  /// \brief make the parameter type accessible
  typedef T parameter_t;

  /// \brief Default constructor
  ParameterBlockSized()
  {
    for (int i = 0; i < dimension; i++)
      parameters_[i] = 0;
  }

  /// \brief Trivial destructor
  virtual ~ParameterBlockSized(){}

  /// @brief Set the values for this parameter block
  /// @param[in] vals The values to copy to this block
  virtual void SetEstimate(const parameter_t& values) = 0;

  /// @brief Set the exact parameter of this parameter block
  /// @param[in] parameters The parameters to set this to
  virtual void SetParameters(const double* parameters)
  {
    if (parameters == NULL)
      LOG(FATAL) << "given pointer is null";
    memcpy(parameters_, parameters, dimension * sizeof(double));
  }

  /// @brief Get the estimote
  /// \return The estimate
  virtual parameter_t GetEstimate() const = 0;

  /// @brief Get pointer to the parameters
  /// \return Pointer to the parameters
  virtual double* GetParameters() 
  {
    return parameters_;
  }

  /// @brief Get pointer to the parameters
  /// \return Pointer to the parameters
  virtual const double* GetParameters() const
  {
    return parameters_;
  }

  /// @brief Get the dimension of this parameter block
  /// \return The dimension
  virtual size_t GetDimension() const
  {
    return dimension;
  }

  /// @brief Get the minimal dimension of this parameter block
  /// \return The minimal dimension
  virtual size_t GetMinimalDimension() const
  {
    return minimal_dimension;
  }

  /**
    * \brief Generalization of the addition operation
    * @param[in] x0 The variable
    * @param[in] delta The perturbation
    * @param[out] x0_plus_delta The perturbed variable
    */
  virtual void Plus(const double* x0, const double* delta,
    double* x0_plus_delta) const
  {
    Eigen::Map<const Eigen::Matrix<double,dimension,1>> x0_map(x0);
    Eigen::Map<const Eigen::Matrix<double,dimension,1>> delta_map(delta);
    Eigen::Map<Eigen::Matrix<double,dimension,1>> x0_plus_delta_map(x0_plus_delta);
    x0_plus_delta_map = x0_map + delta_map;
  }

  /** \brief The jacobian of the plus operation w.r.t. delta at delta = 0
    * @param[in] x0 The variable
    * @praram[out] jacobian The Jacobian
    */
  virtual void PlusJacobian(const double* x0, double* jacobian) const
  {
    Eigen::Map<Eigen::Matrix<double, dimension, minimal_dimension, 
      Eigen::RowMajor>> J(jacobian);

    J.setZero();
    J.topLeftCorner(minimal_dimension, minimal_dimension).setIdentity();
  }

  /** \brief Generalization of the subtraction operation
    * @param[in] x0 The variable
    * @param[in] x0_plus_delta The perturbed variable
    * @param[out] delta The difference
    */
  virtual void Minus(const double* x0, const double* x0_plus_delta,
    double* delta) const
  {
    Eigen::Map<const Eigen::Matrix<double,dimension,1>> x0_map(x0);
    Eigen::Map<Eigen::Matrix<double,dimension,1>> delta_map(delta);
    Eigen::Map<const Eigen::Matrix<double,dimension,1>> x0_plus_delta_map(x0_plus_delta);

    delta_map = x0_plus_delta_map - x0_map;
  }

  /** \brief Computes the Jacobian from minimal to overparameterized space
    * @param[in] x0 The variable
    * @param[out] jacobian the Jacobian
    */
  virtual void LiftJacobian(const double* x0, double* jacobian) const
  {
    Eigen::Map<Eigen::Matrix<double, dimension, minimal_dimension,
      Eigen::RowMajor>> J(jacobian);

    J.setZero();
    J.topLeftCorner(minimal_dimension, minimal_dimension).setIdentity();
  }
  
protected:
  /// @brief the parameters
  double parameters_[dimension];

};

#endif