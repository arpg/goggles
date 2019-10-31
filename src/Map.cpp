#include <Map.h>
#include <ceres/ordered_groups.h>
#include <MarginalizationCostFunction.h>

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
  int prameterization)
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
      problem_->AddParameterBlock(parameter_block->Parameters(),
                                  parameter_block->Dimension());
      break;
    }
    case Parameterization::Orientation:
    {
      problem_->AddParameterBlock(parameter_block->Parameters(),
                                  parameter_block->Dimension(),
                                  &quaternion_parameterization_);
    }
  }
  return true;
}

bool Map::RemoveParameterBlock(uint64_t parameter_block_id)
{
  if (!ParameterBlockExists(parameter_block_id))
    return false;

  const ResidualBlockCollection res = GetResiduals(parameter_block_id);

  for (size_t i = 0; i < res.size(); i++)
    RemoveResidualBlock(res[i].residual_block_id);

  problem_->RemoveParameterBlock(
    GetParameterBlockPtr(parameter_block_id)->Parameters());
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
}