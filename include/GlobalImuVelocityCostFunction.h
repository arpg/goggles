#pragma once
#ifndef GLOBALIMUVELOCITYCOSTFUNCTION_H_
#define GLOBALIMUVELOCITYCOSTFUNCTION_H_

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <QuaternionParameterization.h>
#include <ErrorInterface.h>
#include "DataTypes.h"
#include <ceres/ceres.h>

class GlobalImuVelocityCostFunction : public ceres::CostFunction, 
                                      public ErrorInterface
{
public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<double,12,12> covariance_t;
  typedef covariance_t information_t;
  typedef covariance_t jacobian_t;
  //typedef ceres::SizedCostFunction<12,4,3,3,3,4,3,3,3> base_t;

  /** 
    * @brief Constructor
    * @param[in] t0 Start time
    * @param[in] t1 End time
    * @param[in] imu_measurements Vector of imu measurements from before t0 to after t1
    * @param[in] imu_parameters Imu noise, bias, and rate parameters
    */
  GlobalImuVelocityCostFunction(const double t0, const double t1,
                                const std::vector<ImuMeasurement> &imu_measurements,
                                const ImuParams &imu_parameters);

  ~GlobalImuVelocityCostFunction();

  /**
    * @brief Propagates orientation, velocity, and biases with given imu measurements
    * @remark This function can be used externally to propagate imu measurements
    * @param[in] imu_measurements Vector of imu measurements from before t0 to after t1
    * @param[in] imu_parameters Imu noise, bias, and rate parameters
    * @param[inout] orientation The starting orientation
    * @param[inout] velocity The starting velocity
    * @param[inout] gyro_bias The starting gyro bias
    * @param[inout] accel_bias The starting accelerometer bias
    * @param[in] t0 The starting time
    * @param[in] t1 The end time
    * @param[out] covariance Covariance for the given starting state
    * @param[out] jacobian Jacobian w.r.t. the starting state
    * @return The number of integration steps
    */
  static int Propagation(const std::vector<ImuMeasurement> &imu_measurements,
                         const ImuParams &imu_parameters,
                         Eigen::Quaterniond &orientation,
                         Eigen::Vector3d &velocity,
                         Eigen::Vector3d &gyro_bias,
                         Eigen::Vector3d &accel_bias,
                         double t0, double t1,
                         covariance_t* covariance = NULL,
                         jacobian_t* jacobian = NULL);

  /**
    * @brief Propagates orientation, velocity, and biases with given imu measurements
    * @param[in] orientation The starting orientation
    * @param[in] gyro_bias The starting gyro bias
    * @param[in] accel_bias The starting accelerometer bias
    */
  int RedoPreintegration(const Eigen::Quaterniond &orientation,
                         const Eigen::Vector3d &velocity,
                         const Eigen::Vector3d &gyro_bias,
                         const Eigen::Vector3d &accel_bias) const;

  /**
    * @brief Set the imu parameters
    */
  void SetImuParameters(const ImuParams &imu_parameters)
  {
    imu_parameters_ = imu_parameters;
  }

  /**
    * @brief Set the imu measurements
    */
  void SetImuMeasurements(const std::vector<ImuMeasurement> imu_measurements)
  {
    imu_measurements_ = imu_measurements;
  }

  /**
    * @brief Set the start time
    */
  void SetT0(double t0)
  {
    t0_ = t0;
  }

  /**
    * @brief Set end time
    */
  void SetT1(double t1)
  {
    t1_ = t1;
  }

  /**
    * @brief Get the imu parameters
    */
  const ImuParams& GetImuParameters() const
  {
    return imu_parameters_;
  }

  /**
    * @brief Get the imu measurements
    */
  const std::vector<ImuMeasurement>& GetImuMeasurements() const
  {
    return imu_measurements_;
  }

  /**
    * @brief Get the starting time
    */
  double GetT0() const
  {
    return t0_;
  }

  /**
    * @brief Get the ending time
    */
  double GetT1() const
  {
    return t1_;
  }

  /**
    * @brief Evaluate the error term and compute jacobians
    * @param parameters Pointer to the parameters
    * @param residuals Pointer to the residuals
    * @param jacobians Pointer to the jacobians
    * @return Success of the evaluation
    */
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const;

  /**
    * @brief Evaluate the error term and compute jacobians in both full and minimal form
    * @param parameters Pointer to the parameters
    * @param residuals Pointer to the residuals
    * @param jacobians Pointer to the jacobians
    * @param jacobians_minimal Pointer to the minimal jacobians
    * @return Success of the evaluation
    */
  bool EvaluateWithMinimalJacobians(
      double const* const* parameters, 
      double* residuals, 
      double** jacobians,
      double** jacobians_minimal) const;

  size_t ResidualDim() const;

protected:

  double t0_;
  double t1_;
  std::vector<ImuMeasurement> imu_measurements_;
  ImuParams imu_parameters_;

  // data structures for pre-integration
  mutable std::mutex preintegration_mutex_;
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0);
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();

  // may not be necessary for velocity-only
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulation
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();

  // may not be necessary for velocity-only
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();

  // the jacobian of the increment without biases
  mutable jacobian_t P_delta_ = jacobian_t::Zero();

  // reference biases that are updated in RedoPreintegration
  mutable Eigen::Vector3d velocity_ref_ = Eigen::Vector3d::Zero();
  mutable Eigen::Vector3d gyro_bias_ref_ = Eigen::Vector3d::Zero();
  mutable Eigen::Vector3d accel_bias_ref_ = Eigen::Vector3d::Zero();

  mutable bool redo_ = true; // does RedoPreintegration need to be called?
  mutable int redo_counter_ = 0; // number of times RedoPreintegration is called

  mutable information_t information_;
  mutable information_t sqrt_information_;
};

#endif