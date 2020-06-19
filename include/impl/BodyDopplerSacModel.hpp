
#include <pcl/sample_consensus/eigen.h>
#include <pcl/common/concatenate.h>
#include <BodyDopplerSacModel.h>

#ifndef IMPL_BODYDOPPLERSACMODEL_H_
#define IMPL_BODYDOPPLERSACMODEL_H_

template <typename PointT>
void pcl::BodyDopplerSacModel<PointT>::getSamplePoints(
  std::vector<Eigen::Vector3d> &points, 
  std::vector<double> &dopplers) const
{
  for (int i = 0; i < indices_->size(); i++)
  {
    Eigen::Vector3d p(input_->points[(*indices_)[i]].x,
                      input_->points[(*indices_)[i]].y,
                      input_->points[(*indices_)[i]].z);
    points.push_back(p.normalized());
    dopplers.push_back(input_->points[(*indices_)[i]].doppler);
  }
}

template <typename PointT> 
bool pcl::BodyDopplerSacModel<PointT>::isSampleGood(
  const std::vector<int> &samples) const
{
  // get the xyz points and normalize
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < samples.size(); i++)
  {
    Eigen::Vector3d p(input_->points[samples[i]].x,
                      input_->points[samples[i]].y,
                      input_->points[samples[i]].z);
    points.push_back(p.normalized());
  }

  Eigen::Matrix3d M;
  for (int i = 0; i < points.size(); i++)
  {
    for (int j = 0; j < 3; j++)
      M(i,j) = points[i][j];
  }
  
  Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(M);

  if (lu_decomp.rank() > 2)
  {
    return true;
  }
  else
  {
    return false;
  }
}

template <typename PointT> 
bool pcl::BodyDopplerSacModel<PointT>::computeModelCoefficients (
  const std::vector<int> &samples, 
  Eigen::VectorXf &model_coefficients) const
{
  if (samples.size() != sample_size_)
  {
    PCL_ERROR ("[pcl::BodyDopplerSacModel::computeModelCoefficients] Invalid set of samples given (%lu)!\n", samples.size());
    return false;
  }

  model_coefficients.resize(sample_size_);

  // get the xyz points and normalize
  std::vector<Eigen::Vector3d> eigen_points;
  for (int i = 0; i < samples.size(); i++)
  {
    Eigen::Vector3d p(input_->points[samples[i]].x,
                      input_->points[samples[i]].y,
                      input_->points[samples[i]].z);
    eigen_points.push_back(p.normalized());
  }

  // set up and solve uniquely determined 3 target problem
  Eigen::Matrix3d M;
  Eigen::Vector3d b;
  for (int i = 0; i < eigen_points.size(); i++)
  {
    for (int j = 0; j < 3; j++)
      M(i,j) = eigen_points[i][j];

    b(i) = input_->points[samples[i]].doppler;
  }
  model_coefficients = -1.0 * (M.completeOrthogonalDecomposition().solve(b)).cast<float>();
  return true;
}

template <typename PointT>
void pcl::BodyDopplerSacModel<PointT>::getDistancesToModel(
  const Eigen::VectorXf &model_coefficients,
  std::vector<double> &distances) const
{
  if (!isModelValid(model_coefficients))
  {
    distances.clear();
    return;
  }
  distances.resize(indices_->size());

  // get points and normalize
  std::vector<Eigen::Vector3d> points;
  std::vector<double> dopplers;
  getSamplePoints(points, dopplers);

  // iterate over points and calculate residuals
  for (int i = 0; i < indices_->size(); i++)
  {
    distances[i] = std::fabs((-1.0 * model_coefficients.dot(
      points[i].cast<float>())) - dopplers[i]);
  }
}

template <typename PointT>
void pcl::BodyDopplerSacModel<PointT>::selectWithinDistance(
  const Eigen::VectorXf &model_coefficients,
  const double threshold, 
  std::vector<int> &inliers)
{
  if (!isModelValid(model_coefficients))
  {
    inliers.clear();
    return;
  }
  int nr_p = 0;
  inliers.resize(indices_->size());
  error_sqr_dists_.resize(indices_->size());

  // get points and normalize
  std::vector<Eigen::Vector3d> points;
  std::vector<double> dopplers;
  getSamplePoints(points, dopplers);

  for (int i = 0; i < indices_->size(); i++)
  {
    double distance = std::fabs((-1.0 * model_coefficients.dot(
      points[i].cast<float>())) - dopplers[i]);

    if (distance < threshold)
    {
      inliers[nr_p] = (*indices_)[i];
      error_sqr_dists_[nr_p] = distance;
      nr_p++;
    }
  }

  inliers.resize(nr_p);
  error_sqr_dists_.resize(nr_p);

}

template <typename PointT> 
std::size_t pcl::BodyDopplerSacModel<PointT>::countWithinDistance (
  const Eigen::VectorXf &model_coefficients,
  const double threshold) const
{
  if (!isModelValid(model_coefficients))
    return 0;

  int nr_p = 0;
  std::vector<double> distances;
  getDistancesToModel(model_coefficients, distances);

  for (int i = 0; i < distances.size(); i++)
    if (distances[i] < threshold) nr_p++;

  return nr_p++;
}

template <typename PointT>
void pcl::BodyDopplerSacModel<PointT>::optimizeModelCoefficients(
  const std::vector<int> &inliers, 
  const Eigen::VectorXf &model_coefficients,
  Eigen::VectorXf &optimized_coefficients) const
{
  optimized_coefficients = model_coefficients;
}


template <typename PointT> 
bool pcl::BodyDopplerSacModel<PointT>::doSamplesVerifyModel(
  const std::set<int> &indices, 
  const Eigen::VectorXf &model_coefficients,
  const double threshold) const
{
  if (model_coefficients.size() != model_size_)
  {
    PCL_ERROR ("[pcl::BodyDopplerSacModel::doSamplesVerifyModel] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size ());
    return false;
  }

  // get points and normalize
  std::vector<Eigen::Vector3d> points;
  std::vector<double> dopplers;
  getSamplePoints(points, dopplers);

  for (const int &index : indices)
  {
    double distance = (-1.0 * model_coefficients.dot(
      points[index].cast<float>())) - dopplers[index];
    if (std::fabs(distance) > threshold)
    {
      return false;
    }
  }
  return true;
}

template <typename PointT> 
bool pcl::BodyDopplerSacModel<PointT>::isModelValid(
  const Eigen::VectorXf &model_coefficients) const
{
  if (!SampleConsensusModel<PointT>::isModelValid(model_coefficients))
    return false;

  return true;
}

template <typename PointT>
void pcl::BodyDopplerSacModel<PointT>::projectPoints(
  const std::vector<int> &inliers,
  const Eigen::VectorXf &model_coefficients,
  PointCloud &projected_points,
  bool copy_data_fields) const
{

}

#endif
