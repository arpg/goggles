/**
  * This code borrows heavily from Map.hpp in OKVIS:
  * https://github.com/ethz-asl/okvis/blob/master/okvis_ceres/include/okvis/ceres/ImuError.hpp
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
#ifndef MAP_H_
#define MAP_H_

#include <memory>
#include <ceres/ceres.h>
#include <ParameterBlock.h>
#include <QuaternionParameterization.h>
#include <unordered_map>
#include <ErrorInterface.h>
#include <DataTypes.h>

class Map
{
public:

  /// @brief Constructor
  Map();

  /// @brief Struct to store residual block info
  struct ResidualBlockInfo
  {
    ResidualBlockInfo()
      : residual_block_id(0),
        loss_function_ptr(0),
        error_interface_ptr(std::shared_ptr<ErrorInterface>()) {}

    ResidualBlockInfo(ceres::ResidualBlockId residual_block_id,
      ceres::LossFunction* loss_function_ptr,
      std::shared_ptr<ErrorInterface> error_interface_ptr)
      : residual_block_id(residual_block_id),
        loss_function_ptr(loss_function_ptr),
        error_interface_ptr(error_interface_ptr) {}

    ceres::ResidualBlockId residual_block_id;
    ceres::LossFunction* loss_function_ptr;
    std::shared_ptr<ErrorInterface> error_interface_ptr;
  };

  // associates parameter block id to the parameter block container
  typedef std::pair<uint64_t, std::shared_ptr<ParameterBlock>> ParameterBlockInfo;

  // collection of residual and parameter block containers
  typedef std::vector<ResidualBlockInfo> ResidualBlockCollection;
  typedef std::vector<ParameterBlockInfo> ParameterBlockCollection;

  /// @brief enumerated parameterization types
  enum Parameterization
  {
    Orientation,
    Trivial
  };

  /** @brief Check whether the given parameter block is part of the problem
    * @param parameter_block_id ID of the parameter block
    * @return True if the parameter block is in the map
    */
  bool ParameterBlockExists(uint64_t parameter_block_id) const;
  /*
  /// @brief Log information on a parameter block
  void PrintParameterBlockInfo(uint64_t parameter_block_id) const;

  /// @brief Log information on a residual block
  void PrintResidualBlockInfo(ceres::ResidualBlockId residual_block_id) const;
  */
  /** @brief Add a parameter block to the map
    * @param parameter_block Parameter block to insert
    * @param parameterization The local parameterization
    * @return True if successful
    */
  bool AddParameterBlock(std::shared_ptr<ParameterBlock> parameter_block,
    int parameterization = Parameterization::Trivial);

  /** @brief Remove a parameter block from the map
    * @param parameter_block_id ID of the block to remove
    * @return True if successful
    */
  bool RemoveParameterBlock(uint64_t parameter_block_id);

  /** @brief Remove a parameter block from the map
    * @param parameter_block_ptr Pointer to the parameter block to be removed
    * @return True if successful
    */
  bool RemoveParameterBlock(std::shared_ptr<ParameterBlock> parameter_block);

  /** @brief Add a residual block to the map
    * @param[in] cost_function the cost function 
    * @param[in] loss_function The loss function
    * @param[in] parameter_block_ptrs A vector containing all related parameter blocks
    * @return Id of the residual block
    */
  ceres::ResidualBlockId AddResidualBlock(
    std::shared_ptr<ceres::CostFunction> cost_function,
    ceres::LossFunction* loss_function,
    std::vector<std::shared_ptr<ParameterBlock>>& parameter_block_ptrs);

  /** @brief Add a residual block id, See documentation in ceres problem.h
  * @param[in] cost_function The error term to be used.
  * @param[in] loss_function Use an m-estimator? NULL, if not needed.
  * @param[in] x0 The first parameter block.
  * @param[in] x1 The second parameter block (if existent).
  * @param[in] x2 The third parameter block (if existent).
  * @param[in] x3 The 4th parameter block (if existent).
  * @param[in] x4 The 5th parameter block (if existent).
  * @param[in] x5 The 6th parameter block (if existent).
  * @param[in] x6 The 7th parameter block (if existent).
  * @param[in] x7 The 8th parameter block (if existent).
  * @param[in] x8 The 9th parameter block (if existent).
  * @param[in] x9 The 10th parameter block (if existent).
  * @return The residual block ID, i.e. what cost_function points to.
  */
  ceres::ResidualBlockId AddResidualBlock(
      std::shared_ptr<ceres::CostFunction> cost_function,
      ceres::LossFunction* loss_function,
      std::shared_ptr<ParameterBlock> x0,
      std::shared_ptr<ParameterBlock> x1 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x2 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x3 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x4 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x5 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x6 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x7 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x8 = std::shared_ptr<ParameterBlock>(),
      std::shared_ptr<ParameterBlock> x9 = std::shared_ptr<ParameterBlock>());

  /** @brief Replace the parameters connected to a residual block id
    * @param[in] residual_block_id The ID of the residual block
    * @param[in] parameter_block_ptrs A vector containing the parameter blocks to be replaced
    */
  void ResetResidualBlock(
    ceres::ResidualBlockId residual_block_id,
    std::vector<std::shared_ptr<ParameterBlock>>& parameter_block_ptrs);

  /** @brief Remove a residual block
    * @param[in] id The residual block ID to be removed
    * @return True on success
    */
  bool RemoveResidualBlock(ceres::ResidualBlockId residual_block_id);

  /** @brief Set a parameter block constant
    * @param[in] parameter_block_id the ID of the block to set constant
    * @return True on success
    */
  bool SetParameterBlockConstant(uint64_t parameter_block_id);

  /** @brief Set a parameter block variable
    * @param[in] parameter_block_id the ID of the block to set variable
    * @return True on success
    */
  bool SetParameterBlockVariable(uint64_t parameter_block_id);

  /** @brief Do not optimise a certain parameter block.
    * @param[in] parameter_block Pointer to the parameter block that should be constant.
    * @return True on success.
    */
  bool SetParameterBlockConstant(std::shared_ptr<ParameterBlock> parameter_block) 
  {
    return SetParameterBlockConstant(parameter_block->GetId());
  }

  /** @brief Optimise a certain parameter block (this is the default).
    * @param[in] parameter_block Pointer to the parameter block that should be optimised.
    * @return True on success.
    */
  bool SetParameterBlockVariable(std::shared_ptr<ParameterBlock> parameter_block)
  {
    return SetParameterBlockVariable(parameter_block->GetId());
  }

  /** @brief Set the local parameterization of a parameter block
    * @param[in] parameter_block_id The ID of the parameter block
    * @param[in] local_parameterization Pointer to the parameterization
    */
  bool ResetParameterization(uint64_t parameter_block_id,
    ceres::LocalParameterization* local_parameterization);

  /** @brief Set the local parameterization of a parameter block
    * @param[in] parameter_block_id The ID of the parameter block
    * @param[in] parameterization int indicating the parameterization
    */
  bool ResetParameterization(uint64_t parameter_block_id,
    int parameterization);

  /** @brief Set the local parameterization of a parameter block
    * @param[in] parameter_block Pointer to the parameter block
    * @param[in] local_parameterization Pointer to the parameterization
    * @return True on success
    */
  bool ResetParameterization(std::shared_ptr<ParameterBlock> parameter_block,
    ceres::LocalParameterization* local_parameterization)
  {
    return ResetParameterization(parameter_block->GetId(), local_parameterization);
  }

  /// @brief Get a pointer to the parameter block container
  std::shared_ptr<ParameterBlock> GetParameterBlockPtr(
    uint64_t parameter_block_id);

  /// @brief Get a pointer to the parameter block container
  std::shared_ptr<const ParameterBlock> GetParameterBlockPtr(
    uint64_t parameter_block_id) const;

  /// @brief Get a pointer to a residual
  std::shared_ptr<ErrorInterface> GetErrorInterfacePtr(
    ceres::ResidualBlockId residual_block_id);

  /// @brief Get a pointer to a residual
  std::shared_ptr<const ErrorInterface> GetErrorInterfacePtr(
    ceres::ResidualBlockId residual_block_id) const;

  /// @brief Get the residual blocks associated to a parameter block
  ResidualBlockCollection GetResidualBlocks(uint64_t parameter_block_id) const;

  /// @brief Get the parameter blocks associated to a residual block
  ParameterBlockCollection GetParameterBlocks(
    ceres::ResidualBlockId residual_block_id) const;

  /// \brief Map from parameter block id to parameter block container
  typedef std::unordered_map<uint64_t,
    std::shared_ptr<ParameterBlock>> IdToParameterBlockMap;

  /// \brief Map from parameter block id to residual block info
  typedef std::unordered_map<ceres::ResidualBlockId, ResidualBlockInfo> 
    ResidualBlockIdToInfoMap;

  /// @brief Get map connecting parameter block ids to containers
  const IdToParameterBlockMap& GetIdToParameterBlockMap() const
  {
    return id_to_parameter_block_map_;
  }

  /// @brief Get the map from residual block id to info
  const ResidualBlockIdToInfoMap& GetResidualBlockIdToInfoMap() const
  {
    return residual_block_id_to_info_map_;
  }

  /// \brief Ceres options
  ceres::Solver::Options options;

  /// \brief Ceres optimization summary
  ceres::Solver::Summary summary;

  /// \brief Solve the ceres problem
  void Solve()
  {
    ceres::Solve(options, problem_.get(), &summary);
  }

  /// \brief Get estimated covariance for a set of parameter blocks
  /// \param[in] parameters The set of parameters for which to estimate covariance
  /// \param[out] covariance_matrix The covariance matrix
  bool GetCovariance(std::vector<std::shared_ptr<ParameterBlock>> &parameters,
                     Eigen::MatrixXd &covariance_matrix);


protected:

  /// \brief the ceres problem
  std::shared_ptr<ceres::Problem> problem_;

  /// \brief Map from parameter block id to residual block info
  typedef std::unordered_multimap<uint64_t, ResidualBlockInfo> IdToResidualBlockMultiMap;

  /// \brief Map from residual block id to its parameter blocks
  typedef std::unordered_map<ceres::ResidualBlockId, 
    ParameterBlockCollection> ResidualBlockIdToParameterBlockCollectionMap;

  /// \brief The map connectning praameter block ids to parameter blocks
  IdToParameterBlockMap id_to_parameter_block_map_;

  /// \brief The map from residual id to residual block info
  ResidualBlockIdToInfoMap residual_block_id_to_info_map_;

  /// \brief The map from parameter block id to residual block pointer
  IdToResidualBlockMultiMap id_to_residual_block_multimap_;

  /// \brief The map from residual block id to its associated parameter blocks
  ResidualBlockIdToParameterBlockCollectionMap residual_block_id_to_parameter_block_collection_map_;

  /// \brief Store parameterization locally
  QuaternionParameterization quaternion_parameterization_;
};

#endif