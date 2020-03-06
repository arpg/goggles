#pragma once
#ifndef HOMOGENEOUSPOINTPARAMETERBLOCK_H_
#define HOMOGENEOUSPOINTPARAMETERBLOCK_H_

#include<ParameterBlockSized.h>
#include<HomogeneousPointParameterization.h>
#include<Eigen/Core>

class HomogeneousPointParameterBlock : public ParameterBlockSized <4,3,Eigen::Vector4d>
{
public:
  typedef Eigen::Vector4d estimate_t;
  typedef ParameterBlockSized<4,3,Eigen::Vector4d> base_t;

  HomogeneousPointParameterBlock();

  HomogeneousPointParameterBlock(const Eigen::Vector4d &point, uint64_t id);

  virtual void SetEstimate(const Eigen::Vector4d &point);

  virtual Eigen::Vector4d GetEstimate() const;

  virtual std::string GetTypeInfo() const
  {
    return "HomogeneousPointParameterBlock";
  }

  void SetInitialized(bool initialized) {initialized_ = initialized;}

  bool IsInitialized() const {return initialized_;}

  bool AddObservation(double t);

  std::vector<double> GetObservations() {return observation_times_;}

  /** \brief Generalization of the plus operation
    * @param[in] x0 The variable
    * @param[in] delta The perturbation
    * @param[out] x0_plus_delta the perturbed variable
    */
  virtual void Plus (const double* x0, 
                     const double* delta, 
                     double* x0_plus_delta)
  {
    HomogeneousPointParameterization hp;
    hp.plus(x0, delta, x0_plus_delta);
  }

  /** \brief the jacobian of the plus operation w.r.t. delta at delta = 0
    * @param[in] x0 The variable
    * @param[out] jacobian The jacobian
    */
  virtual void PlusJacobian(const double* x0, double* jacobian) const
  {
    HomogeneousPointParameterization hp;
    hp.ComputeJacobian(x0, jacobian);
  }

  /** \brief Generalization of the minus operation
    * @param[in] x0 The variable
    * @param[in] x0_plus_delta the perturbed variable
    * @param[in] delta The perturbation
    */
  virtual void Minus(const double* x0, 
                     const double* x0_plus_delta, 
                     double* delta) const
  {
    HomogeneousPointParameterization hp;
    hp.Minus(x0, x0_plus_delta, delta);
  }

  /** \brief Computes the jacobian to move from minimal space to the overparameterized space
    * @param[in] x0 Variable
    * @param[out] jacobian the Jacobian
    */
  virtual void LiftJacobian(const double* x0, double* jacobian) const
  {
    HomogeneousPointParameterization hp;
    hp.ComputeLiftJacobian(x0,jacobian);
  }

private:
  bool initialized_;
  std::vector<double> observation_times_;
};

#endif