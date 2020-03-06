#include <HomogeneousPointParameterBlock.h>

HomogeneousPointParameterBlock::HomogeneousPointParameterBlock() 
  : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

HomogeneousPointParameterBlock::HomogeneousPointParameterBlock(
  const Eigen::Vector4d &point, uint64_t id)
{
  SetEstimate(point);
  SetId(id);
  SetFixed(false);
}

void HomogeneousPointParameterBlock::SetEstimate(const Eigen::Vector4d &point)
{
  for (size_t i = 0; i < base_t::dimension; i++)
    parameters_[i] = point[i];
}

Eigen::Vector4d HomogeneousPointParameterBlock::GetEstimate() const
{
  return Eigen::Vector4d(parameters_[0], parameters_[1], parameters_[2],parameters_[3]);
}

bool HomogeneousPointParameterBlock::AddObservation(double t)
{
  if (observation_times_.size() > 0 && t < observation_times_.back())
  {
    LOG(ERROR) << "attempted to add observation in the past, dropping observation";
    return false;
  }
  observation_times_.push_back(t);
  return true;
}