
template<typename Scalar = double>
struct BodyVelocityCostFunction
{
  BodyVelocityCostFunction(const Scalar &doppler,
                           const Eigen::Matrix<Scalar,3,1> target,
                           const double &weight)
      : doppler_(doppler), 
        target_(target),
        weight_(weight) {}
  
  template<typename T>
  bool operator()(const T* const b_v, T* r) const
  {
    CHECK_NOTNULL(b_v);
    CHECK_NOTNULL(r);
    
    // get target velocity as -1 * body velocity
    const Eigen::Map<const Eigen::Matrix<T,3,1>> body_v(b_v);
    Eigen::Matrix<T,3,1> target_v = static_cast<T>(-1.0) * body_v;

    // get unit vector from target to sensor
    Eigen::Matrix<T,3,1> ray_ts = target_.template cast<T>();
    ray_ts.normalize();

    // get projection of body velocity onto target-sensor vector
    T v_r = target_v.dot(ray_ts);
    
    // project target velocity on to ray from sensor to target
    *r = (v_r - static_cast<T>(doppler_)) * static_cast<T>(weight_);
    return true;
  }

  const Scalar doppler_;
  const Eigen::Matrix<Scalar,3,1> target_;
  const double weight_;
};