#include<OrientationParameterBlock.h>

OrientationParameterBlock::OrientationParameterBlock() : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

OrientationParameterBlock::OrientationParameterBlock(
  const Eigen::Quaterniond& orientation, uint64_t id, double timestamp)
{
  SetEstimate(orientation);
  SetId(id);
  SetTimestamp(timestamp);
  SetFixed(false);
}

void OrientationParameterBlock::SetEstimate(const Eigen::Quaterniond &orientation)
{
  for (int i = 0; i < base_t::dimension; i++)
    parameters_[i] = orientation.coeffs()[i];
}

Eigen::Quaterniond OrientationParameterBlock::GetEstimate() const
{
  return Eigen::Quaterniond(parameters_[3], parameters_[0], parameters_[1], parameters_[2]);
}