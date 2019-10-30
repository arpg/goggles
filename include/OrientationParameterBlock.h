#pragma once
#ifndef ORIENTATIONPARAMETERBLOCK_H_
#define ORIENTATIONPARAMETERBLOCK_H_

#include<ParameterBlockSized.h>
#include<QuaternionParameterization.h>
#include<Eigen/Core>

class OrientationParameterBlock : public ParameterBlockSized<4, 3, Eigen::Quaterniond>
{
public:

  /// \brief The estimate type
  typedef Eigen::Quaterniond estimate_t;

  /// \brief The base class type
  typedef ParameterBlockSized<4, 3, Eigen::Quaterniond> base_t;

  /// \brief Default constructor
  OrientationParameterBlock();

  /** \brief Constructor with initial estimate and id
    * @param[in] delta The delta correction to the target ray
    * @param[in] id The unique parameter block id
    * @param[in] timestamp the timestamp of this block
    */
  OrientationParameterBlock(const Eigen::Quaterniond& delta, uint64_t id, double timestamp);

  /// @brief Set the estimate of this block
  /// @param[in] orientation The estimate to set this block to
  virtual void SetEstimate(const Eigen::Quaterniond& orientation);

  /// @brief Set the timestamp of this block
  /// @param[in] timestamp the timestamp of this block
  void SetTimestamp(const double timestamp) {timestamp_ = timestamp;}

  /// @brief Get the estimate
  /// \return The estimate
  virtual Eigen::Quaterniond GetEstimate() const;

  /// @brief Get the time of this parameter block
  /// \return The timestamp of this state
  double GetTimestamp() const {return timestamp_;}

  /// @brief Return parameter block type as a string
  virtual std::string GetTypeInfo() const
  {
    return "OrientationParameterBlock";
  }

  /** \brief Generalization of the plus operation
    * @param[in] x0 The variable
    * @param[in] delta The perturbation
    * @param[out] x0_plus_delta the perturbed variable
    */
  virtual void Plus(const double* x0, const double* delta, 
    double* x0_plus_delta) const
  {
    QuaternionParameterization qp;
    qp.Plus(x0, delta, x0_plus_delta);
  }

  /** \brief the jacobian of the plus operation w.r.t. delta at delta = 0
    * @param[in] x0 The variable
    * @param[out] jacobian The jacobian
    */
  virtual void PlusJacobian(const double* x0, double* jacobian) const
  {
    QuaternionParameterization qp;
    qp.ComputeJacobian(x0, jacobian);
  }

  /** \brief Generalization of the minus operation
    * @param[in] x0 The variable
    * @param[in] x0_plus_delta the perturbed variable
    * @param[in] delta The perturbation
    */
  virtual void Minus(const double* x0, const double* x0_plus_delta, 
    double* delta) const
  {
    QuaternionParameterization qp;
    qp.Minus(x0, x0_plus_delta, delta);
  }

  /** \brief Computes the jacobian to move from minimal space to the overparameterized space
    * @param[in] x0 Variable
    * @param[out] jacobian the Jacobian
    */
  virtual void LiftJacobian(const double* x0, double* jacobian) const
  {
    QuaternionParameterization qp;
    qp.ComputeLiftJacobian(x0,jacobian);
  }

private:
  double timestamp_;

};

#endif