#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <ceres/ceres.h>
#include <BodyVelocityCostFunction.h>
#include <VelocityChangeCostFunction.h>
#include <MarginalizationError.h>
#include <chrono>

struct RadarPoint
{
	PCL_ADD_POINT4D;
	float intensity;
	float range;
	float doppler;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (RadarPoint,
									(float, x, x)
									(float, y, y)
									(float, z, z)
									(float, intensity, intensity)
									(float, range, range)
									(float, doppler, doppler))

typedef pcl::PointCloud<RadarPoint> RadarPointCloud;


class RadarVelocityReader
{
public:

  RadarVelocityReader(ros::NodeHandle nh)
  {
    nh_ = nh;
    std::string radar_topic;
    nh_.getParam("radar_topic", radar_topic);
    VLOG(2) << "radar topic: " << radar_topic;

    pub_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("/mmWaveDataHdl/velocity",1);
    sub_ = nh_.subscribe(radar_topic, 0, &RadarVelocityReader::callback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;

    window_size_ = 4;

    // set up ceres problem
    doppler_loss_ = new ceres::CauchyLoss(0.15);
    //marginalization_scaling_ = new ceres::ScaledLoss(NULL, 0.01, ceres::DO_NOT_TAKE_OWNERSHIP);
    accel_loss_ = NULL;//new ceres::CauchyLoss(0.1);

    ceres::Problem::Options prob_options;

    prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    prob_options.enable_fast_removal = true;
    //solver_options_.check_gradients = true;
    //solver_options_.gradient_check_relative_precision = 1.0e-6;
    solver_options_.num_threads = 4;
    solver_options_.max_num_iterations = 300;
    solver_options_.update_state_every_iteration = true;
    solver_options_.function_tolerance = 1e-10;
    solver_options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    problem_.reset(new ceres::Problem(prob_options));
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
	  double timestamp = msg->header.stamp.toSec();
	  pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
	  pcl::fromROSMsg(*msg, *cloud);

    // Reject clutter
    Declutter(cloud);
    
		for (int i = 0; i < cloud->size(); i++)
		{
			cloud->at(i).y *= -1.0;
			cloud->at(i).z *= -1.0;
		}

    
    if (cloud->size() < 10)
		{
      LOG(ERROR) << "input cloud has less than 10 points, output will be unreliable";
		}

		bool no_doppler = true;
		for (int i = 0; i < cloud->size(); i++)
		{
			if (cloud->at(i).doppler > 0)
				no_doppler = false;
		}
		if (no_doppler)
		{
			LOG(ERROR) << "no doppler reading in current cloud";
		}
		else
		{
			LOG(ERROR) << "getting velocity";
    	geometry_msgs::TwistWithCovarianceStamped vel_out;
    	vel_out.header.stamp = msg->header.stamp;

    	// Get velocity measurements
    	auto start = std::chrono::high_resolution_clock::now();
    	GetVelocityCeres(cloud, timestamp, vel_out);
    	//GetVelocityIRLS(cloud, coeffs, 100, 1.0e-10, vel_out);
    	auto finish = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> elapsed = finish - start;
    	sum_time_ += elapsed.count();
    	num_iter_++;
    	LOG(ERROR) << "execution time: " << sum_time_ / double(num_iter_);
    	pub_.publish(vel_out);
		}
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  ceres::CauchyLoss *doppler_loss_;
  ceres::CauchyLoss *accel_loss_;
  //std::deque<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> velocities_;
  std::deque<Eigen::Vector3d*> velocities_;
  std::deque<double> timestamps_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
  std::shared_ptr<ceres::Problem> problem_;
  //ceres::ScaledLoss *marginalization_scaling_;
  std::shared_ptr<MarginalizationError> marginalization_error_ptr_;
  ceres::ResidualBlockId marginalization_id_;
  ceres::Solver::Options solver_options_;
  int window_size_;
  double min_range_;
  int num_iter_;
  double sum_time_;

  /** \brief Clean up radar point cloud prior to processing
    * \param[in,out] cloud the input point cloud
    */
  void Declutter(pcl::PointCloud<RadarPoint>::Ptr cloud)
  {
    int orig_size = cloud->size();
    RadarPointCloud::iterator it = cloud->begin();
    while (it != cloud->end())
    {
      if (it->range <= min_range_)
        it = cloud->erase(it);
      else
        it++;
    }
  }

  /** \brief Get body velocity using iteratively reweighted linear least squares
    * \param[in] cloud the input point cloud
    * \param[out] vel the resultant body velocity
    * \param[in] max_iter the maximum number of iterations to run IRLS
    * \param[in] func_tol the minimum cost improvement over total cost required
    * to continue iterating
    * \param[out] vel_out the resultant ros message
    * \note this function is currently not used, it was written to demonstrate
    * this process could be done with linear least squares
    */
  void GetVelocityIRLS(pcl::PointCloud<RadarPoint>::Ptr cloud,
                       Eigen::Vector3d &velocity, 
                       int max_iter, double func_tol,
                       geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {

    // assemble model, measurement, and weight matrices
    int N = cloud->size();
    Eigen::MatrixXd X(N,3);
    Eigen::MatrixXd y(N,1);
    Eigen::MatrixXd W(N,N);
    W.setZero();

    // get min and max intensity for calculating weights
    double min_intensity, max_intensity;
    getIntensityBounds(min_intensity,max_intensity,cloud);

    for (int i = 0; i < N; i++)
    {
      // set measurement matrix entry as doppler measurement
      y(i) = cloud->at(i).doppler;

      // set model matrix row as unit vector from sensor to target
      Eigen::Matrix<double, 1, 3> ray_st;
      ray_st << cloud->at(i).x,
                cloud->at(i).y,
                cloud->at(i).z;
      X.block<1,3>(i,0) = ray_st.normalized();

      // set weight as normalized intensity
      W(i,i) = (cloud->at(i).intensity - min_intensity)
              / (max_intensity - min_intensity);
    }

    // initialize weights as normalized intensities
    Eigen::MatrixXd W_p(N,N);
    W_p = W;

    // estimate velocities through iteratively re-weighted least squares
    Eigen::MatrixXd res(N,1);
    Eigen::MatrixXd psi(N,1);
    double sum_res = 1.0e4;
    double delta_res = 1.0;
    double d_cost_over_cost = 1.0;
    int num_iter = 0;
    double c = 1.0 / (0.15*0.15);
    double initial_cost = 0;

    while (num_iter < max_iter
          && d_cost_over_cost > func_tol)
    {
      // get the weighted least squares solution
      velocity = (X.transpose()*W_p*X).inverse()*X.transpose()*W_p*y;

      // evaluate the residual
      res = (X * velocity) - y;

      // compute new weights using the cauchy robust norm
      for (int i = 0; i < N; i++)
      {
        psi(i) = std::max(std::numeric_limits<double>::min(),
                      1.0 / (1.0 + res(i)*res(i) * c));
        double w_i = psi(i);// / res(i);
        W_p(i,i) = W(i,i) * w_i;
      }

      // compute sum residual with new weights
      Eigen::MatrixXd new_sum_res = res.transpose() * W_p*W_p * res;
      delta_res = fabs(sum_res - new_sum_res(0));
      sum_res = new_sum_res(0);
      d_cost_over_cost = delta_res / sum_res;
      if (num_iter == 0) initial_cost = sum_res;
      num_iter++;
    }

    VLOG(3) << "initial cost: " << initial_cost;
    VLOG(3) << "final cost:   " << sum_res;
    VLOG(3) << "change:       " << initial_cost - sum_res;
    VLOG(3) << "iterations:   " << num_iter;

    velocity = velocity * -1.0;

    // get estimate covariance
    double sum_psi_sq = 0.0;
    double sum_psi_p = 0.0;

    for (int i = 0; i < N; i++)
    {
      sum_psi_sq += psi(i) * psi(i) * W_p(i,i);
      double inv = 1.0 / (1.0 + res(i)*res(i) * c);
      sum_psi_p += -1.0 * c * inv * inv * W_p(i,i);
    }
    Eigen::MatrixXd cov = sum_psi_sq / (sum_psi_p * sum_psi_p)
                      * (X.transpose() * X).inverse();
    Eigen::Matrix3d covariance(cov);

    populateMessage(vel_out, velocity, covariance);
  }

  /** \brief uses ceres solver to estimate the body velocity from the input
    * point cloud
    * \param[out] the resultant velocity and covariance in ros message form
    */
  void GetVelocityCeres(pcl::PointCloud<RadarPoint>::Ptr cloud, 
                   double timestamp,
                   geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {
    // add latest parameter block and remove old one if necessary
    if (velocities_.size() == 0)
		{
			velocities_.push_front(new Eigen::Vector3d());
			velocities_.front()->setZero();
		}
		else
		{
			velocities_.push_front(new Eigen::Vector3d(*velocities_.front()));
		}
		std::vector<ceres::ResidualBlockId> residuals;
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);
    problem_->AddParameterBlock(velocities_.front()->data(),3);
    if (velocities_.size() > window_size_)
    {
      // remove marginalization error from problem if it's
      // already initialized
      if (marginalization_error_ptr_ && marginalization_id_)
      {
        LOG(INFO) << "removing marginalization residual";
        problem_->RemoveResidualBlock(marginalization_id_);
        marginalization_id_ = 0;
      }

      // if the marginalization error has not been initialized
      // initialize it
      if (!marginalization_error_ptr_)
      {
        LOG(INFO) << "resetting marginalization error";
        marginalization_error_ptr_.reset(
          new MarginalizationError(problem_));
      }

      // add last state and associated residuals to marginalization error
      LOG(INFO) << "adding states and residuals to marginalization";
      if (!marginalization_error_ptr_->AddResidualBlocks(residual_blks_.back()))
        LOG(ERROR) << "failed to add residuals";
      std::vector<double*> state_to_marginalize;
      state_to_marginalize.push_back(velocities_.back()->data());
      if (!marginalization_error_ptr_->MarginalizeOut(state_to_marginalize))
        LOG(ERROR) << "failed to marginalize state";
      marginalization_error_ptr_->UpdateErrorComputation();

      LOG(INFO) << "deleting old bookkeeping";
      
      velocities_.pop_back();
      residual_blks_.pop_back();
      timestamps_.pop_back();

      // discard marginalization error if it has no residuals
      if (marginalization_error_ptr_->num_residuals() == 0)
      {
        LOG(INFO) << "no residuals associated to marginalization";
        marginalization_error_ptr_.reset();
      }

      // add marginalization term back to the problem
      if (marginalization_error_ptr_)
      {
        LOG(INFO) << "adding marginalization error to problem";
        std::vector<double*> parameter_block_ptrs;
        marginalization_error_ptr_->GetParameterBlockPtrs(
          parameter_block_ptrs);
        marginalization_id_ = problem_->AddResidualBlock(
          marginalization_error_ptr_.get(), 
          NULL,
          parameter_block_ptrs);
      }
      LOG(INFO) << "done with marginalization";
    }
    
    // calculate uniform weighting for doppler measurements
    double weight = 100.0 / cloud->size();

    // add residuals on doppler readings
    for (int i = 0; i < cloud->size(); i++)
    {
      Eigen::Vector3d target(cloud->at(i).x,
                             cloud->at(i).y,
                             cloud->at(i).z);
      ceres::CostFunction* doppler_cost_function =
        new BodyVelocityCostFunction(cloud->at(i).doppler,
                                      target,
                                      weight);
      // add residual block to ceres problem
      ceres::ResidualBlockId res_id = 
					problem_->AddResidualBlock(doppler_cost_function, 
                                     doppler_loss_, 
                                     velocities_.front()->data());

      residual_blks_.front().push_back(res_id);
    }

    // add residual on change in velocity if applicable
    if (timestamps_.size() >= 2)
    {
      double delta_t = timestamps_[0] - timestamps_[1];
      ceres::CostFunction* vel_change_cost_func =
        new VelocityChangeCostFunction(delta_t);

      ceres::ResidualBlockId res_id = 
						problem_->AddResidualBlock(vel_change_cost_func, 
                                       accel_loss_, 
                                       velocities_[1]->data(), 
                                       velocities_[0]->data());

      residual_blks_[1].push_back(res_id);
    }
    // solve the ceres problem and get result
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);
    LOG(INFO) << summary.FullReport();
    LOG(INFO) << "velocity from ceres: " << velocities_.front()->transpose();

    // get estimate covariance
    /*
    ceres::Covariance::Options cov_options;
    cov_options.num_threads = 1;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(cov_options);

    std::vector<std::pair<const double*, const double*>> cov_blks;
    cov_blks.push_back(std::make_pair(velocities_.front()->data(),
                                      velocities_.front()->data()));

    covariance.Compute(cov_blks, problem_.get());
    Eigen::Matrix3d covariance_matrix;
    covariance.GetCovarianceBlock(velocities_.front()->data(),
                                  velocities_.front()->data(),
                                  covariance_matrix.data());

    VLOG(2) << "covariance: \n" << covariance_matrix;
    */
    // Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
		Eigen::Matrix3d covariance_matrix;
		// Eigen::DiagonalMatrix<double, 3> covariance_matrix;
		covariance_matrix.diagonal() << 0.01, 0.015, 0.05;

    populateMessage(vel_out,*(velocities_.front()),covariance_matrix);
  }

  /** \brief populate ros message with velocity and covariance
    * \param[out] vel the resultant ros message
    * \param[in] velocity the estimated velocity
    * \param[in] covariance the estimate covariance
    */
  void populateMessage(geometry_msgs::TwistWithCovarianceStamped &vel,
                        Eigen::Vector3d &velocity,
                        Eigen::Matrix3d &covariance)
  {
		vel.header.frame_id = "base_radar_link";

    vel.twist.twist.linear.x = velocity[0];
    vel.twist.twist.linear.y = velocity[1];
    vel.twist.twist.linear.z = velocity[2];

		vel.twist.covariance[0] = covariance(0,0);
		vel.twist.covariance[7] = covariance(1,1);
		vel.twist.covariance[14] = covariance(2,2);

    // for (int i = 0; i < 3; i++)
    // {
    //   for (int j = 0; j < 3; j++)
    //     vel.twist.covariance[(i*6)+j] = covariance(i,j);
    // }
  }

  /** \brief get the maximum and minimum intensity in the input cloud
    * \param[out] min_intensity the minimum intensity found in the cloud
    * \param[out] max_intensity the maximum intensity found in the cloud
    * \param[in] the input point cloud
    */
  void getIntensityBounds(double &min_intensity,
                          double &max_intensity,
                          pcl::PointCloud<RadarPoint>::Ptr cloud)
  {
    min_intensity = 1.0e4;
    max_intensity = 0.0;
    for (int i = 0; i < cloud->size(); i++)
    {
      if (cloud->at(i).intensity > max_intensity)
        max_intensity = cloud->at(i).intensity;
      if (cloud->at(i).intensity < min_intensity)
        min_intensity = cloud->at(i).intensity;
    }
  }
};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
	ros::init(argc, argv, "radar_vel");
  ros::NodeHandle nh("~");
  RadarVelocityReader* rv_reader = new RadarVelocityReader(nh);

  ros::spin();

  return 0;
}
