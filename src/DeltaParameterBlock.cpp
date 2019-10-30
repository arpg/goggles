#include<DeltaParameterBlock.h>

DeltaParameterBlock::DeltaParameterBlock() : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

DeltaParameterBlock::DeltaParameterBlock(
  const Eigen::Vector3d& delta, uint64_t id, double timestamp)
{
  SetEstimate(delta);
  SetId(id);
  SetTimestamp(timestamp);
  SetFixed(false);
}

void DeltaParameterBlock::SetEstimate(const Eigen::Vector3d &delta)
{
  for (int i = 0; i < base_t::dimension; i++)
    parameters_[i] = delta[i];
}

Eigen::Vector3d DeltaParameterBlock::GetEstimate() const
{
  return Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]);
}