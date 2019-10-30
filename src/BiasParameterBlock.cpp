#include<BiasParameterBlock.h>

BiasParameterBlock::BiasParameterBlock() : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

BiasParameterBlock::BiasParameterBlock(
  const Eigen::Vector3d& bias, uint64_t id, double timestamp)
{
  SetEstimate(bias);
  SetId(id);
  SetTimestamp(timestamp);
  SetFixed(false);
}

void BiasParameterBlock::SetEstimate(const Eigen::Vector3d &bias)
{
  for (int i = 0; i < base_t::dimension; i++)
    parameters_[i] = bias[i];
}

Eigen::Vector3d BiasParameterBlock::GetEstimate() const
{
  return Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]);
}