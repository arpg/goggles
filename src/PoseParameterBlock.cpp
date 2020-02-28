#include<PoseParameterBlock.h>

PoseParameterBlock::PoseParameterBlock() : base_t::ParameterBlockSized()
{
  SetFixed(false);
}

PoseParameterBlock::~PoseParameterBlock() {}

PoseParameterBlock::PoseParameterBlock(
  const Transformation& T_WS, uint64_t id, double timestamp)
{
  SetEstimate(T_WS);
  SetId(id);
  SetTimestamp(timestamp);
  SetFixed(false);
}

void PoseParameterBlock::SetEstimate(const Transformation &T_WS)
{
  const Eigen::Vector3d r = T_WS.r();
  const Eigen::Vector4d q = T_WS.q().coeffs();
  parameters_[0] = r[0];
  parameters_[1] = r[1];
  parameters_[2] = r[2];
  parameters_[3] = q[0];
  parameters_[4] = q[1];
  parameters_[5] = q[2];
  parameters_[6] = q[3];
}

Transformation PoseParameterBlock::GetEstimate() const
{
  return Transformation(
    Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]),
    Eigen::Quaterniond(parameters_[6], parameters_[3], 
                       parameters_[4], parameters_[5]));
}