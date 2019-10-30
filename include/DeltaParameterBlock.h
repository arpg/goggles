#pragma once
#ifndef DELTAPARAMETERBLOCK_H_
#define DELTAPARAMETERBLOCK_H_

#include<ParameterBlockSized.h>
#include<Eigen/Core>

class DeltaParameterBlock : public ParameterBlockSized<3, 3, Eigen::Vector3d>
{
public:

  /// \brief The estimate type
  typedef Eigen::Vector3d estimate_t;

  /// \brief The base class type
  typedef ParameterBlockSized<3, 3, Eigen::Vector3d> base_t;

  /// \brief Default constructor
  DeltaParameterBlock();

  /** \brief Constructor with initial estimate and id
    * @param[in] delta The delta correction to the target ray
    * @param[in] id The unique parameter block id
    */
  DeltaParameterBlock(const Eigen::Vector3d& delta, uint64_t id, double timestamp);

  /// @brief Set the estimate of this block
  /// @param[in] delta The estimate to set this block to
  virtual void SetEstimate(const Eigen::Vector3d& delta);

  /// @brief Set the timestamp of this block
  /// @param[in] timestamp the timestamp of this block
  void SetTimestamp(const double timestamp) {timestamp_ = timestamp;}

  /// @brief Get the estimate
  /// \return The estimate
  virtual Eigen::Vector3d GetEstimate() const;

  /// @brief Get the time of this parameter block
  /// \return The timestamp of this state
  double GetTimestamp() const {return timestamp_;}

  /// @brief Return parameter block type as a string
  virtual std::string GetTypeInfo() const
  {
    return "DeltaParameterBlock";
  }

private:
  double timestamp_;

};

#endif