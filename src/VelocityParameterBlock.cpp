#include<VelocityParameterBlock.h>

VelocityParameterBlock::VelocityParameterBlock() : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

VelocityParameterBlock::VelocityParameterBlock(
  const Eigen::Vector3d& velocity, uint64_t id, double timestamp)
{
  SetEstimate(velocity);
  SetId(id);
  SetTimestamp(timestamp);
  SetFixed(false);
}

void VelocityParameterBlock::SetEstimate(const Eigen::Vector3d &velocity)
{
  for (int i = 0; i < base_t::dimension; i++)
    parameters_[i] = velocity[i];
}

Eigen::Vector3d VelocityParameterBlock::GetEstimate() const
{
  return Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]);
}