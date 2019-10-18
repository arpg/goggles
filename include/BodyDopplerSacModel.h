#pragma once

#ifndef BODYDOPPLERSACMODEL_H_
#define BODYDOPPLERSACMODEL_H_

#include <pcl/sample_consensus/sac_model.h>

namespace pcl
{
  class BodyDopplerSacModel : public SampleConsensusModel<RadarPoint>
  {
  public:
    using SampleConsensusModel<RadarPoint>::model_name_;
    using SampleConsensusModel<RadarPoint>::input_;
    using SampleConsensusModel<RadarPoint>::indices_;
    using SampleConsensusModel<RadarPoint>::radius_min_;
    using SampleConsensusModel<RadarPoint>::radius_max_;
    using SampleConsensusModel<RadarPoint>::error_sqr_dists_;

    using PointCloud = typename SampleConsensusModel<RadarPoint>::PointCloud;
    using PointCloudPtr = typename SampleConsensusModel<RadarPoint>::PointCloudPtr;
    using PointCloudConstPtr = typename SampleConsensusModel<RadarPoint>::PointCloudConstPtr;

    using Ptr = boost::shared_ptr<BodyDopplerSacModel<RadarPoint> >;

    /** \brief Constructor
      * \param[in] cloud the input point cloud dataset
      * \param[in] random if true set the random seed to the current time (default: false)
      */
    BodyDopplerSacModel(const PointCloudConstPtr &cloud, bool random = false)
      : SampleConsensusModel<RadarPoint>(cloud, random)
    {
      model_name_ = "BodyDopplerSacModel";
      sample_size_ = 3;
      model_size_ = 3;
    }

    /** \brief Constructor
      * \param[in] cloud the input point cloud dataset
      * \param[in] indices a vector of point indices to be used from the cloud
      * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
      */
    BodyDopplerSacModel(const PointCloudConstPtr &cloud,
                        const std::vector<int> &indices,
                        bool random = false)
      : SampleConsensusModel<RadarPoint> (cloud, indices, random)
    {
      model_name_ = "BodyDopplerSacModel";
      sample_size_ = 3;
      model_size_ = 3;
    }

    /** \brief Copy constructor
      * \param[in] source the model to copy into this
      */
    BodyDopplerSacModel (const BodyDopplerSacModel &source)
      : SampleConsensusModel<RadarPoint> ()
    {
      *this = source;
      model_name_ = "BodyDopplerSacModel";
    }

    /** \brief Empty destructor */
    ~BodyDopplerSacModel() {}

    /** \brief Copy constructor
      * \param[in] source model to copy into this
      */
    inline BodyDopplerSacModel&
    operator = (const BodyDopplerSacModel &source)
    {
      SampleConsensusModel<RadarPoint>::operator=(source);
      return (*this);
    }

    /** \brief Check whether the given index samples can form a valid
      * doppler body velocity model, compute the model coefficients
      * from these samples and store them in model_coefficients.
      * The model coefficients are v_x, v_y, and v_z.
      * \param[in] samples the point indices found as possible candidates
      * \param[out] model_coefficients the resultant model coefficients
      */
    bool computeModelCoefficients (const std::vector<int> &samples,
                                   Eigen::VectorXf &model_coefficients) const override;

    /** \brief Compute all distances from the cloud data to a given doppler body model
      * \param[in] model_coefficients the coefficients of the body doppler model
      * \param[out] distances the resultant estimated distances
      */
    void getDistancesToModel (const Eigen::VectorXf &model_coefficients,
                              std::vector<double> &distances) const override;

    /** \brief Compute all distances from the cloud data to a given model
      * \param[in] model_coefficients the coefficients of the body doppler model
      * \param[in] threshold a maximum admissible distance threshold for 
      * determining the inlier set
      * \param[out] inliers the resultant inlier set
      */
    void selectWithinDistance (const Eigen::VectorXf &model_coefficients,
                               const double threshold,
                               std::vector<int> &inliers) override;

    /** \brief Count all points which are inliers for the given model coefficients
      * \param[in] model_coefficients the coefficients of the body doppler model
      * \param[in] threshold maximum admissible distance threshold for inliers
      * \return the resultant number of inliers
      */
    int countWithinDistance (const Eigen::VectorXf *model_coefficients,
                             const double threshold) const override;

    /** \brief Recompute the body doppler model coefficients using the given inlier set
      * \param[in] inliers the inlier points found as supporting the model
      * \param[in] model_coefficients the initial guess for optimization
      * \param[out] optimized_coefficients the resultant optimized coefficients
      */
    void optimizeModelCoefficients (const std::vector<int> &inliers,
                                    const Eigen::VectorXf &model_coefficients,
                                    Eigen::VectorXf &optimized_coefficients) const override;

    /** \brief Create a new point cloud with inliers projected to the body doppler model
      * \param[in] inliers the data inliers that we want to project
      * \param[in] model_coefficients the coefficients of the body doppler model
      * \param[out] projected_points the resultant projected points
      * \param[in] copy_data_fields set to true if we need to copy the other data fields
      */
    void projectPoints(const std::std::vector<int> &inliers,
                       const Eigen::VectorXf &model_coefficients,
                       PointCloud &projected_points,
                       bool copy_data_fields = true) const override;

    /** \brief Verify whether a subset of indices verifies the given model coefficients
      * \param[in] indices the data indices that need to be tested against the model
      * \param[in] model_coefficients the body doppler model coefficients
      * \praam[in] threshold a maximum admissible distance threshold
      */
    bool doSamplesVerifyModel (const std::set<int> &indices,
                               const Eigen::VectorXf &model_coefficients,
                               const double threshold) const override;

    /** \brief Return a unique id for this model */
    inline pcl::SacModel getModelType() const override {return (SACMODEL_BODYDOPPLER)};

  protected:
    using SampleConsensusModel<RadarPoint>::sample_size_;
    using SampleConsensusModel<RadarPoint>::model_size_;

    /** \brief Check whether a model is valid given the user constraints
      * \param[in] model_coefficents the set of model coefficients
      */
    bool isModelValid (const Eigen::VectorXf &model_coefficients) const override;

    /** \brief Check if a sample of indices results in a good sample of indices
      * \param[in] samples the resultant index samples
      */
    bool isSampleGood (const std::vector<int> &samples) const override;

  private:
    
    /** \brief Functor for the optimization function */
    struct OptimizationFunctor : pcl::Functor<float>
    {
      /** \brief Functor constructor
        * \param[in] estimator pointer to the estimator object
        * \param[in] indices the indices of data points to evaluate
        */
      OptimizationFunctor (const pcl::BodyDopplerSacModel *model,
                           const std::vector<int> &indices)
        : pcl::Functor<float> (indices.size()), model_(model), indices_(indices) {}

      /** \brief Cost function to be minimized
        * \param[in] x the variables array
        * \param[out] fvec the resultant function evaluations
        * \return 0
        */
      int operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
      {
        for (int i = 0; i < values(); i++)
        {

        }
      }
      const pcl::BodyDopplerSacModel *model_;
      const std::vector<int> &indices_;
    };

  };
}

#include <implementation/BodyDopplerSacModel.hpp>
#endif