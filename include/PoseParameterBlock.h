/**
  * This code borrows heavily from PoseParameterBlock.hpp in OKVIS:
  * https://github.com/ethz-asl/okvis/blob/master/okvis_ceres/include/okvis/ceres/PoseParameterBlock.hpp
  * persuant to the following copyright:
  *
  *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
  *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
  *
  *  Redistribution and use in source and binary forms, with or without
  *  modification, are permitted provided that the following conditions are met:
  * 
      * Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.
      * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
        its contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.
  */

#pragma once
#ifndef POSEPARAMETERBLOCK_H_
#define POSEPARAMETERBLOCK_H_

#include <ParameterBlockSized.h>
#include <PoseParameterization.h>
#include <Eigen/Core>
#include <Transformation.h>

class PoseParameterBlock: public ParameterBlockSized<7, 6, Transformation>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The estimate type
  typedef Transformation estimate_t;

  /// \brief Base class type
  typedef ParameterBlockSized<7,6,Transformation> base_t;

  /// \brief Default constructor
  PoseParameterBlock();

  /// \brief Constructor with estimate and time
  /// @param[in] T_WS The pose estimate
  /// @param[in] id The ID of this block
  /// @param[in] timestamp The timestamp of this block
  PoseParameterBlock(const Transformation &T_WS, uint64_t id, const double timestamp);

  /// \brief Trivial destructor
  virtual ~PoseParameterBlock();

  //setters

  /// \brief Set the estimate of this prarameter block
  /// @param[in] T_WS The estimate
  virtual void SetEstimate(const Transformation &T_WS);

  /// \brief Set the timestamp
  /// @param[in] timestamp The timestamp
  void SetTimestamp(const double timestamp) {timestamp_ = timestamp;}

  /// \brief Get the estimate
  /// \return The estimate
  virtual Transformation GetEstimate() const;

  /// \brief Get the time of this block
  /// \return The timestamp of this block
  double GetTimestamp() const {return timestamp_;}

  /// \brief Return the parameter block type as a string
  virtual std::string GetTypeInfo() const
  {
    return "PoseParameterBlock";
  }

  /// \brief Generalization of the plus operation
  /// @param[in] x0 The variable
  /// @param[in] delta The perturbation
  /// @param[out] x0_plus_delta the perturbed variable
  virtual void Plus(const double* x0, const double* delta,
    double* x0_plus_delta) const
  {
    PoseParameterization p;
    p.Plus(x0,delta,x0_plus_delta);
  }

  /// \brief Jacobian of Plus(x, delta) w.r.t. delta at delta = 0
  /// @param[in] x0 The variable
  /// @param[out] jacobian The jacobian
  virtual void PlusJacobian(const double* x0, double* jacobian) const
  {
    PoseParameterization p;
    p.plusJacobian(x0, jacobian);
  }

  /// \brief Generalization of the minus operation
  /// @param[in] x0 The variable
  /// @param[in] x0_plus_delta The perturbed variable
  /// @parab[out] delta The difference
  virtual void Minus(const double* x0, 
                     const double* x0_plus_delta, 
                     double* delta) const
  {
    PoseParameterization p;
    p.Minus(x0, x0_plus_delta, delta);
  }

  /// \brief Computes the jacobian from minimal to overparameterized space
  /// @param[in] x0 The variable
  /// @param[out] jacobian The jacobian
  virtual void LiftJacobian(const double* x0, double* jacobian) const
  {
    PoseParameterization p;
    p.liftJacobian(x0,jacobian);
  }

private:
  double timestamp_;
};


#endif