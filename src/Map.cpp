/**
  * This code borrows heavily from Map.cpp in OKVIS:
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

#include <Map.h>
#include <ceres/ordered_groups.h>

Map::Map()
{
  ceres::Problem::Options options;

  options.local_parameterization_ownership =
    ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

  options.loss_function_ownership = 
    ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

  options.cost_function_ownership = 
    ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

  problem_.reset(new ceres::Problem(options));
}

bool Map::ParameterBlockExists(uint64_t parameter_block_id) const
{
  return (id_to_parameter_block_map_.find(parameter_block_id)
    != id_to_parameter_block_map_.end());
}

bool Map::AddParameterBlock(std::shared_ptr<ParameterBlock> parameter_block, 
  int parameterization)
{
  if (ParameterBlockExists(parameter_block->GetId()))
    return false;

  id_to_parameter_block_map_.insert(
    std::pair<uint64_t, std::shared_ptr<ParameterBlock>> (
      parameter_block->GetId(), parameter_block));

  switch (parameterization)
  {
    case Parameterization::Trivial:
    {
      problem_->AddParameterBlock(parameter_block->GetParameters(),
                                  parameter_block->GetDimension());
      break;
    }
    case Parameterization::Orientation:
    {
      problem_->AddParameterBlock(parameter_block->GetParameters(),
                                  parameter_block->GetDimension(),
                                  &quaternion_parameterization_);
      break;
    }
    case Parameterization::HomogeneousPoint:
    {
      problem_->AddParametBlock(parameter_block->GetParameters(),
                                parameter_block->GetDimension(),
                                &homogeneous_point_parameterization_);
      break;
    }
    case Parameterization::Pose:
    {
      problem_->AddParametBlock(parameter_block->GetParameters(),
                                parameter_block->GetDimension(),
                                &pose_parameterization_);
      break;
    }
  }
  return true;
}

bool Map::RemoveParameterBlock(uint64_t parameter_block_id)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  const ResidualBlockCollection res = GetResidualBlocks(parameter_block_id);

  for (size_t i = 0; i < res.size(); i++)
    RemoveResidualBlock(res[i].residual_block_id);

  problem_->RemoveParameterBlock(
    GetParameterBlockPtr(parameter_block_id)->GetParameters());
  id_to_parameter_block_map_.erase(parameter_block_id);

  return true;
}

bool Map::RemoveParameterBlock(std::shared_ptr<ParameterBlock> parameter_block)
{
  return RemoveParameterBlock(parameter_block->GetId());
}

ceres::ResidualBlockId Map::AddResidualBlock(
  std::shared_ptr<ceres::CostFunction> cost_function,
  ceres::LossFunction* loss_function,
  std::vector<std::shared_ptr<ParameterBlock>>& parameter_block_ptrs)
{
  ceres::ResidualBlockId return_id;
  std::vector<double*> parameter_blocks;
  ParameterBlockCollection parameter_block_collection;

  // find parameter blocks and ids for all parameter blocks
  for (size_t i = 0; i < parameter_block_ptrs.size(); i++)
  {
    parameter_blocks.push_back(parameter_block_ptrs[i]->GetParameters());

    parameter_block_collection.push_back(
      ParameterBlockInfo(parameter_block_ptrs[i]->GetId(),
        parameter_block_ptrs[i]));
  }

  // add residual in ceres problem
  return_id = problem_->AddResidualBlock(cost_function.get(), loss_function,
    parameter_blocks);

  // update bookkeeping
  std::shared_ptr<ErrorInterface> error_interface_ptr = 
    std::dynamic_pointer_cast<ErrorInterface>(cost_function);

  // ensure the cost function implements the error interface
  if (error_interface_ptr == NULL)
    LOG(FATAL) << "cost function does not implement error interface";

  // associate residual id to residual info container
  residual_block_id_to_info_map_.insert(
    std::pair<ceres::ResidualBlockId, ResidualBlockInfo>(
      return_id,
      ResidualBlockInfo(return_id, loss_function, error_interface_ptr)));

  // associate residual block id to parameter blocks to which it's connected
  std::pair<ResidualBlockIdToParameterBlockCollectionMap::iterator, bool> result =
    residual_block_id_to_parameter_block_collection_map_.insert(
      std::pair<ceres::ResidualBlockId, ParameterBlockCollection>(
        return_id, parameter_block_collection));

  if (!result.second)
    return ceres::ResidualBlockId(0);

  // update residual block pointers on associated parameter blocks
  for (uint64_t parameter_id = 0;
    parameter_id < parameter_block_collection.size(); parameter_id++)
  {
    id_to_residual_block_multimap_.insert(
      std::pair<uint64_t, ResidualBlockInfo>(
        parameter_block_collection[parameter_id].first,
        ResidualBlockInfo(return_id, loss_function, error_interface_ptr)));
  }
  return return_id;
}

ceres::ResidualBlockId Map::AddResidualBlock(
  std::shared_ptr<ceres::CostFunction> cost_function,
  ceres::LossFunction* loss_function,
  std::shared_ptr<ParameterBlock> x0,
  std::shared_ptr<ParameterBlock> x1,
  std::shared_ptr<ParameterBlock> x2,
  std::shared_ptr<ParameterBlock> x3,
  std::shared_ptr<ParameterBlock> x4,
  std::shared_ptr<ParameterBlock> x5,
  std::shared_ptr<ParameterBlock> x6,
  std::shared_ptr<ParameterBlock> x7,
  std::shared_ptr<ParameterBlock> x8,
  std::shared_ptr<ParameterBlock> x9)
{
  std::vector<std::shared_ptr<ParameterBlock>> parameter_block_ptrs;

  if (x0)
    parameter_block_ptrs.push_back(x0);
  if (x1)
    parameter_block_ptrs.push_back(x1);
  if (x2)
    parameter_block_ptrs.push_back(x2);
  if (x3)
    parameter_block_ptrs.push_back(x3);
  if (x4)
    parameter_block_ptrs.push_back(x4);
  if (x5)
    parameter_block_ptrs.push_back(x5);
  if (x6)
    parameter_block_ptrs.push_back(x6);
  if (x7)
    parameter_block_ptrs.push_back(x7);
  if (x8)
    parameter_block_ptrs.push_back(x8);
  if (x9)
    parameter_block_ptrs.push_back(x9);

  return Map::AddResidualBlock(cost_function, loss_function, parameter_block_ptrs);
}

void Map::ResetResidualBlock(
  ceres::ResidualBlockId residual_block_id,
  std::vector<std::shared_ptr<ParameterBlock>>& parameter_block_ptrs)
{
  // get residual block info
  ResidualBlockInfo info =
    residual_block_id_to_info_map_[residual_block_id];

  // remove residual from old parameter set
  ResidualBlockIdToParameterBlockCollectionMap::iterator it =
    residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);

  if (it == residual_block_id_to_parameter_block_collection_map_.end())
    LOG(FATAL) << "residual block not in map";

  // iterate over parameter blocks associated to this residual block
  for (ParameterBlockCollection::iterator parameter_it = it->second.begin();
    parameter_it != it->second.end(); parameter_it++)
  {
    // get the id of the parameter block
    uint64_t parameter_id = parameter_it->second->GetId();

    // get all the residuals associated to this parameter block
    std::pair<IdToResidualBlockMultiMap::iterator,
      IdToResidualBlockMultiMap::iterator> range = 
        id_to_residual_block_multimap_.equal_range(parameter_id);

    if (range.first == id_to_residual_block_multimap_.end())
      LOG(FATAL) << "invalid range found";

    // go over the residual blocks and remove if it's being reset
    for (IdToResidualBlockMultiMap::iterator it2 = range.first; 
      it2 != range.second;)
    {
      if (residual_block_id == it2->second.residual_block_id)
        it2 = id_to_residual_block_multimap_.erase(it2);
      else
        it2++;
    }
  }

  // put all new parameter blocks into a vector with id -> parameter block relation
  ParameterBlockCollection parameter_block_collection;
  for (size_t i = 0; i < parameter_block_ptrs.size(); i++)
  {
    parameter_block_collection.push_back(
      ParameterBlockInfo(parameter_block_ptrs.at(i)->GetId(),
        parameter_block_ptrs.at(i)));
  }

  // associate new parameter block collection with the residual we're resetting
  it->second = parameter_block_collection;

  // update residual block pointers
  for (uint64_t parameter_id = 0;
    parameter_id < parameter_block_collection.size(); parameter_id++)
  {
    id_to_residual_block_multimap_.insert(
      std::pair<uint64_t, ResidualBlockInfo>(
        parameter_block_collection[parameter_id].first, info));
  }
}

bool Map::RemoveResidualBlock(ceres::ResidualBlockId residual_block_id)
{
  // remove block in ceres problem
  problem_->RemoveResidualBlock(residual_block_id);

  // find in our residual->parameters map
  ResidualBlockIdToParameterBlockCollectionMap::iterator it =
    residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);

  if (it == residual_block_id_to_parameter_block_collection_map_.end())
    return false;

  // iterate over parameter blocks associated to this residual
  for (ParameterBlockCollection::iterator parameter_it = it->second.begin();
    parameter_it != it->second.end(); parameter_it++)
  {
    uint64_t parameter_id = parameter_it->second->GetId();

    // get all residuals associated to this parameter block
    std::pair<IdToResidualBlockMultiMap::iterator,
      IdToResidualBlockMultiMap::iterator> range = 
        id_to_residual_block_multimap_.equal_range(parameter_id);

    if (range.first == id_to_residual_block_multimap_.end())
      LOG(FATAL) << "invalid range found";

    // remove associations
    for (IdToResidualBlockMultiMap::iterator it2 = range.first;
      it2 != range.second;)
    {
      if (residual_block_id == it2->second.residual_block_id)
        it2 = id_to_residual_block_multimap_.erase(it2);
      else
        it2++;
    }
  }

  // update other bookkeeping
  residual_block_id_to_parameter_block_collection_map_.erase(it);
  residual_block_id_to_info_map_.erase(residual_block_id);
  return true;
}

bool Map::SetParameterBlockConstant(uint64_t parameter_block_id)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  std::shared_ptr<ParameterBlock> parameter_block = 
    id_to_parameter_block_map_.find(parameter_block_id)->second;

  parameter_block->SetFixed(true);
  problem_->SetParameterBlockConstant(parameter_block->GetParameters());
  return true;
}

bool Map::SetParameterBlockVariable(uint64_t parameter_block_id)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  std::shared_ptr<ParameterBlock> parameter_block =
    id_to_parameter_block_map_.find(parameter_block_id)->second;

  parameter_block->SetFixed(false);
  problem_->SetParameterBlockVariable(parameter_block->GetParameters());
  return true;
}

bool Map::ResetParameterization(uint64_t parameter_block_id,
  int parameterization)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  // can't change parameterization so we need to remove the parameter
  // block and re-add it with a new parameterization
  ResidualBlockCollection res = GetResidualBlocks(parameter_block_id);
  std::shared_ptr<ParameterBlock> parameter_block_ptr = GetParameterBlockPtr(
    parameter_block_id);

  // get pointers to all parameters associated to the residuals
  std::vector<std::vector<std::shared_ptr<ParameterBlock>>> parameter_block_ptrs(
    res.size());

  for (size_t r = 0; r < res.size(); r++)
  {
    ParameterBlockCollection p_info = GetParameterBlocks(res[r].residual_block_id);

    for (size_t p = 0; p < p_info.size(); p++)
      parameter_block_ptrs[r].push_back(p_info[p].second);
  }

  // remove the parameter block
  // this causes the removal of associated residuals
  RemoveParameterBlock(parameter_block_id);

  // add the parameter block back with new parameterization
  AddParameterBlock(parameter_block_ptr, parameterization);

  // re-add all residual blocks
  for (size_t r = 0; r < res.size(); r++)
  {
    AddResidualBlock(
      std::dynamic_pointer_cast<ceres::CostFunction>(
        res[r].error_interface_ptr),
      res[r].loss_function_ptr, parameter_block_ptrs[r]);
  }

  return true;
}

bool Map::ResetParameterization(
  uint64_t parameter_block_id,
  ceres::LocalParameterization* local_parameterization)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  problem_->SetParameterization(
    id_to_parameter_block_map_.find(parameter_block_id)->second->GetParameters(),
    local_parameterization);

  id_to_parameter_block_map_.find(parameter_block_id)->second
    ->SetLocalParameterization(local_parameterization);

  return true;
}

std::shared_ptr<ParameterBlock> 
Map::GetParameterBlockPtr(uint64_t parameter_block_id)
{
  if (!ParameterBlockExists(parameter_block_id))
    return std::shared_ptr<ParameterBlock>();

  return id_to_parameter_block_map_.find(parameter_block_id)->second;
}

std::shared_ptr<const ParameterBlock> 
Map::GetParameterBlockPtr(uint64_t parameter_block_id) const
{
  if (!ParameterBlockExists(parameter_block_id))
    return std::shared_ptr<const ParameterBlock>();

  return id_to_parameter_block_map_.find(parameter_block_id)->second;
}

Map::ResidualBlockCollection Map::GetResidualBlocks(uint64_t parameter_block_id) const
{
  IdToResidualBlockMultiMap::const_iterator it1 = 
    id_to_residual_block_multimap_.find(parameter_block_id);

  // return empty collection if parameter block id is not in the problem
  if (it1 == id_to_residual_block_multimap_.end())
    return Map::ResidualBlockCollection();

  ResidualBlockCollection return_residuals;

  // get the range of all elements associated to the parameter block id
  std::pair<IdToResidualBlockMultiMap::const_iterator,
    IdToResidualBlockMultiMap::const_iterator> range = 
    id_to_residual_block_multimap_.equal_range(parameter_block_id);

  // go through range and add residual blocks to return vector
  for (IdToResidualBlockMultiMap::const_iterator it = range.first;
    it != range.second; it++)
    return_residuals.push_back(it->second);

  return return_residuals;
}

std::shared_ptr<ErrorInterface> Map::GetErrorInterfacePtr(
  ceres::ResidualBlockId residual_block_id)
{
  ResidualBlockIdToInfoMap::iterator it = 
    residual_block_id_to_info_map_.find(residual_block_id);

  if (it == residual_block_id_to_info_map_.end())
    return std::shared_ptr<ErrorInterface>();

  return it->second.error_interface_ptr;
}

std::shared_ptr<const ErrorInterface> Map::GetErrorInterfacePtr(
  ceres::ResidualBlockId residual_block_id) const
{
  ResidualBlockIdToInfoMap::const_iterator it = 
    residual_block_id_to_info_map_.find(residual_block_id);

  if (it == residual_block_id_to_info_map_.end())
    return std::shared_ptr<ErrorInterface>();

  return it->second.error_interface_ptr;
}

Map::ParameterBlockCollection Map::GetParameterBlocks(
  ceres::ResidualBlockId residual_block_id) const
{
  ResidualBlockIdToParameterBlockCollectionMap::const_iterator it =
    residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);

  if (it == residual_block_id_to_parameter_block_collection_map_.end())
  {
    ParameterBlockCollection empty;
    return empty;
  }  
  return it->second;
}

bool Map::GetCovariance(std::vector<std::shared_ptr<ParameterBlock>> &parameters,
                     Eigen::MatrixXd &covariance_matrix)
{
  size_t total_dim = 0;
  for (size_t i = 0; i < parameters.size(); i++)
    total_dim += parameters[i]->GetDimension();

  covariance_matrix.resize(total_dim,total_dim);
  covariance_matrix.setZero();

  ceres::Covariance::Options cov_options;
  cov_options.num_threads = 4;
  cov_options.algorithm_type = ceres::DENSE_SVD;
  cov_options.null_space_rank = -1;
  ceres::Covariance covariance(cov_options);

  std::vector<std::pair<const double*, const double*>> cov_blks;

  for (size_t i = 0; i < parameters.size(); i++)
  {
    for (size_t j = i; j < parameters.size(); j++)
    {
      cov_blks.push_back(std::make_pair(parameters[i]->GetParameters(),
                                        parameters[j]->GetParameters()));
    }
  }

  bool result = covariance.Compute(cov_blks, problem_.get());

  int idx_i = 0;
  for (size_t i = 0; i < parameters.size(); i++)
  {
    int idx_j = idx_i;
    const int i_dim = parameters[i]->GetDimension();

    for (size_t j = i; j < parameters.size(); j++)
    {
      const int j_dim = parameters[j]->GetDimension();
      Eigen::MatrixXd block_ij(i_dim,j_dim);
      covariance.GetCovarianceBlock(parameters[i]->GetParameters(),
                                    parameters[j]->GetParameters(),
                                    block_ij.data());
      covariance_matrix.block(idx_i,idx_j,i_dim,j_dim) = block_ij;
      covariance_matrix.block(idx_j,idx_i,j_dim,i_dim) = block_ij.transpose();
      idx_j += j_dim;
    }
    idx_i += i_dim;
  }

  // copy upper triangle to lower

  return result;
}