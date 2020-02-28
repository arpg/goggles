/**
  * This code borrows heavily from Transformation.hpp in OKVIS:
  * https://github.com/ethz-asl/okvis/blob/master/okvis_ceres/include/okvis/kinematics/Transformation.hpp
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
#ifndef TRANSFORMATION_H_
#define TRANSFORMATION_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

double sinc(double x);
Eigen::Matrix3d crossMx(Eigen::Vector3d);
Eigen::Quaterniond deltaQ(const Eigen::Vector3d &dAlpha);
Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &PhiVec);

class Transformation
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Default constructor: initializes a unit transform
  Transformation();

  /// \brief Copy constructor
  Transformation(const Transformation &other);

  /// \brief Move constructor
  Transformation(Transformation && other);

  /// \brief Construct from translation vector and orientation quaternion
  /// @param[in] r_AB The translation r_AB 
  /// @param[in] q_AB The orientation quaternion q_AB
  Transformation(const Eigen::Vector3d &r_AB, const Eigen::Quaterniond &q_AB);

  /// \brief Construct from a transformation matrix
  /// @param[in] T_AB The transformation matrix
  explicit Transformation(const Eigen::Matrix4d &T_AB);

  /// \brief Destructor
  ~Transformation();

  /// \brief Set coefficients
  /// \tparam Derived_coeffs Deducible matrix type
  /// @param[in] coeffs The parameters [r_AB,q_AB], q_AB as [x,y,z,w]
  template<typename Derived_coeffs>
  bool setCoeffs(const Eigen::MatrixBase<Derived_coeffs> &coeffs);

  /// \brief The underlying transformation matrix
  Eigen::Matrix4d T() const;

  /// \brief Returns the orientation as a 3x3 rotation matrix
  const Eigen::Matrix3d &C() const;

  /// \brief Returns the translation vector r_AB 
  const Eigen::Map<Eigen::Vector3d> &r() const;

  /// \brief Returns the orientation quaternion q_AB
  const Eigen::Map<Eigen::Quaterniond> &q() const;

  /// \brief Get the upper 3x4 part of the transformation matrix T_AB
  Eigen::Matrix<double,3,4> T3x4() const;

  /// \brief Return the coefficients as [r_AB,q_AB], with q_AB as [x,y,z,w]
  const Eigen::Matrix<double,7,1> &coeffs() const
  {
    return coefficients_;
  }

  /// \brief get pointer to the coefficients
  const double* coeffPtr() const
  {
    return &coefficients_[0];
  }

  /// \brief Set to a random transformation
  void setRandom();

  /// \brief Set to a random transformation with bounded rotation and translation
  /// @param[in] maxTranslation Maximum translation [m]
  /// @param[out] maxRotation Maximum rotation [rad]
  void setRandom(double maxTranslation, double maxRotation);

  /// \brief set from a 4x4 transformation matrix
  /// @param[in] T_AB The transformation matrix
  void set(const Eigen::Matrix4d &T_AB);

  /// \brief set from a translation vector and rotation quaternion
  /// @param[in] r_AB The translation
  /// @param[in] q_AB The orientation
  void set(const Eigen::Vector3d &r_AB, const Eigen::Quaterniond &q_AB);

  /// \brief Set to identity
  void setIdentity();

  /// \brief Get an identity transformation
  static Transformation Identity();

  /// \brief Returns a copy of the transformation inverted
  Transformation inverse() const;

  /// \brief Multiplication with another transformation object (group operator)
  /// @param[in] rhs The right-hand side transformation
  Transformation operator*(const Transformation &rhs) const;

  /// \brief Rotate a direction vector 
  /// @param[in] rhs The right-hand side direction
  Eigen::Vector3d operator*(const Eigen::Vector3d &rhs) const;

  /// \brief Transform a homogeneous point
  /// @param rhs The right-hand side point
  Eigen::Vector4d operator*(const Eigen::Vector4d &rhs) const;

  /// \brief Assignment -- copy
  /// @param[in] rhs The rhs for this to be assigned to
  Transformation & operator=(const Transformation &rhs);

  Eigen::Matrix4d qplus(const Eigen::Quaterniond & q_AB) const;

  /// \brief Apply a small update with delta being 6x1
  /// \tparam Derived_delta Deducible matrix type
  /// @param[in] delta the 6x1 minimal update
  /// \return True on success
  template<typename Derived_delta>
  bool oplus(const Eigen::MatrixBase<Derived_delta> &delta);

  /// \brief Apply a small update with delta being 6x1
  /// @param[in] delta the 6x1 minimal update
  /// @param[out] jacobian The output Jacobian
  /// \return True on success
  template<typename Derived_delta, typename Derived_jacobian>
  bool oplus(const Eigen::MatrixBase<Derived_delta> &delta,
    const Eigen::MatrixBase<Derived_jacobian> &jacobian);

  /// \brief Get the jacobian of the oplus operation
  /// @param[out] jacobian The output jacobian
  /// \return True on success
  template<typename Derived_jacobian>
  bool oplusJacobian(const Eigen::MatrixBase<Derived_jacobian> &jacobian) const;

  /// \brief Gets jacobian dx/dChi
  ///     i.e. lift the minimal jacobian to a full one
  /// @param[out] jacobian The output lift jacobian
  /// \return True on success
  template<typename Derived_jacobian>
  bool liftJacobian(const Eigen::MatrixBase<Derived_jacobian> &jacobian) const;

protected:
  void updateC();
  Eigen::Matrix<double,7,1> coefficients_; ///< Concatenated coefficients [r;q]
  Eigen::Map<Eigen::Vector3d> r_; ///< Translation {_A}r_{B}
  Eigen::Map<Eigen::Quaterniond> q_; ///< Quaternion q_{AB}
  Eigen::Matrix3d C_; ///< The cached DCM C_{AB}
};

#include "impl/Transformation.hpp"

#endif