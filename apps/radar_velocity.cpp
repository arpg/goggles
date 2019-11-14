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
#include <DataTypes.h>
#include <BodyVelocityCostFunction.h>
#include <VelocityChangeCostFunction.h>
#include <MarginalizationError.h>
#include <IdProvider.h>
#include <Map.h>
#include <VelocityParameterBlock.h>
#include <DeltaParameterBlock.h>
#include <pcl/sample_consensus/impl/mlesac.hpp>
#include <BodyDopplerSacModel.h>
#include <chrono>


class RadarVelocityReader
{
public:

  RadarVelocityReader(ros::NodeHandle nh) : map_ptr_(new Map())
  {
    nh_ = nh;
    std::string radar_topic;
    nh_.getParam("radar_topic", radar_topic);
    nh_.getParam("publish_inliers", publish_inliers_);
    VLOG(2) << "radar topic: " << radar_topic;

		// get node namespace
    std::string ns = ros::this_node::getNamespace();

    vel_est_publisher_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
      ns + "/mmWaveDataHdl/velocity",1);
    if (publish_inliers_)
      inlier_set_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
        ns + "/mmWaveDataHdl/inlier_set",1);
    sub_ = nh_.subscribe(radar_topic, 0, &RadarVelocityReader::callback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;

    window_size_ = 4;

    // set up ceres problem
    //map_ptr_->options.check_gradients = true;
    doppler_loss_ = new ceres::CauchyLoss(1.0);
    //marginalization_scaling_ = new ceres::ScaledLoss(NULL, 0.01, ceres::DO_NOT_TAKE_OWNERSHIP);
    accel_loss_ = NULL;//new ceres::CauchyLoss(0.1);
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
    vel_est_publisher_.publish(vel_out);
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher vel_est_publisher_;
  ros::Publisher inlier_set_publisher_;
  bool publish_inliers_;
  ros::Subscriber sub_;
  ceres::LossFunction *doppler_loss_;
  ceres::LossFunction *accel_loss_;

  std::deque<std::shared_ptr<VelocityParameterBlock>> velocities_;
  std::deque<std::vector<std::shared_ptr<DeltaParameterBlock>>> ray_errors_;
  std::deque<double> timestamps_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
  std::shared_ptr<Map> map_ptr_;

  std::shared_ptr<MarginalizationError> marginalization_error_ptr_;
  ceres::ResidualBlockId marginalization_id_;
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
  void GetVelocityIRLS(pcl::PointCloud<RadarPoint>::Ptr raw_cloud,
                       Eigen::Vector3d &velocity,
                       int max_iter, double func_tol,
                       geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {

    // remove outliers with mlesac
    pcl::BodyDopplerSacModel<RadarPoint>::Ptr model(
      new pcl::BodyDopplerSacModel<RadarPoint>(raw_cloud));
    std::vector<int> inliers;
    pcl::MaximumLikelihoodSampleConsensus<RadarPoint> mlesac(model,0.10);
    mlesac.computeModel();
    mlesac.getInliers(inliers);

    // copy inlier points to new data structure;
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::copyPointCloud(*raw_cloud, inliers, *cloud);

    // publish inlier set if requested
    if (publish_inliers_)
    {
      sensor_msgs::PointCloud2 inliers_out;
      pcl::PCLPointCloud2 cloud2;
      pcl::toPCLPointCloud2(*cloud, cloud2);
      pcl_conversions::fromPCL(cloud2, inliers_out);
      inlier_set_publisher_.publish(inliers_out);
    }

    // assemble model, measurement, and weight matrices
    int N = cloud->size();
    Eigen::MatrixXd X(N,3);
    Eigen::MatrixXd y(N,1);
    Eigen::MatrixXd W(N,N);
    W.setZero();

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
      W(i,i) = 1.0 / double(N);
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
  void GetVelocityCeres(pcl::PointCloud<RadarPoint>::Ptr raw_cloud,
                   double timestamp,
                   geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {
    
    // remove outliers with mlesac
    pcl::BodyDopplerSacModel<RadarPoint>::Ptr model(
      new pcl::BodyDopplerSacModel<RadarPoint>(raw_cloud));
    std::vector<int> inliers;
    pcl::MaximumLikelihoodSampleConsensus<RadarPoint> mlesac(model,0.10);
    mlesac.computeModel();
    mlesac.getInliers(inliers);

    // copy inlier points to new data structure;
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::copyPointCloud(*raw_cloud, inliers, *cloud);

    // publish inlier set if requested
    if (publish_inliers_)
    {
      sensor_msgs::PointCloud2 inliers_out;
      pcl::PCLPointCloud2 cloud2;
      pcl::toPCLPointCloud2(*cloud, cloud2);
      pcl_conversions::fromPCL(cloud2, inliers_out);
      inlier_set_publisher_.publish(inliers_out);
    }
    
    // add latest parameter block and remove old one if necessary
    uint64_t vel_id = IdProvider::instance().NewId();
    if (velocities_.size() == 0)
		{
			velocities_.push_front(std::make_shared<VelocityParameterBlock>(
        Eigen::Vector3d::Zero(), vel_id, timestamp));
		}
		else
		{
			velocities_.push_front(std::make_shared<VelocityParameterBlock>(
        velocities_.front()->GetEstimate(), vel_id, timestamp));
		}
    std::vector<std::shared_ptr<DeltaParameterBlock>> ray_err;
    ray_errors_.push_front(ray_err);
		std::vector<ceres::ResidualBlockId> residuals;
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);

    map_ptr_->AddParameterBlock(velocities_.front());

    auto start = std::chrono::high_resolution_clock::now(); 
    if (velocities_.size() > window_size_)
    {
      // remove marginalization error from problem if it's
      // already initialized
      if (marginalization_error_ptr_ && marginalization_id_)
      {
        map_ptr_->RemoveResidualBlock(marginalization_id_);
        marginalization_id_ = 0;
      }

      // if the marginalization error has not been initialized
      // initialize it
      if (!marginalization_error_ptr_)
      {
        marginalization_error_ptr_.reset(
          new MarginalizationError(map_ptr_));
      }

      // add last state and associated residuals to marginalization error
      if (!marginalization_error_ptr_->AddResidualBlocks(residual_blks_.back()))
        LOG(ERROR) << "failed to add residuals";

      std::vector<uint64_t> states_to_marginalize;
      states_to_marginalize.push_back(velocities_.back()->GetId());

      for (int i = 0; i < ray_errors_.back().size(); i++)
        states_to_marginalize.push_back(ray_errors_.back()[i]->GetId());

      if (!marginalization_error_ptr_->MarginalizeOut(states_to_marginalize))
        LOG(ERROR) << "failed to marginalize state";

      marginalization_error_ptr_->UpdateErrorComputation();

      velocities_.pop_back();
      ray_errors_.pop_back();
      residual_blks_.pop_back();
      timestamps_.pop_back();

      // discard marginalization error if it has no residuals
      if (marginalization_error_ptr_->num_residuals() == 0)
      {
        LOG(ERROR) << "no residuals associated to marginalization";
        marginalization_error_ptr_.reset();
      }

      // add marginalization term back to the problem
      if (marginalization_error_ptr_)
      {
        std::vector<std::shared_ptr<ParameterBlock>> parameter_block_ptrs;
        marginalization_error_ptr_->GetParameterBlockPtrs(
          parameter_block_ptrs);
        marginalization_id_ = map_ptr_->AddResidualBlock(
          marginalization_error_ptr_,
          NULL,
          parameter_block_ptrs);
      }
      
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    LOG(INFO) << "marginalization time: " << elapsed.count();

    // calculate uniform weighting for doppler measurements
    double weight = 100.0 / cloud->size();
    double d = 1.0;

    // add residuals on doppler readings
    for (int i = 0; i < cloud->size(); i++)
    {
      // create parameter for ray error
      uint64_t ray_id = IdProvider::instance().NewId();
      ray_errors_.front().push_back(std::make_shared<DeltaParameterBlock>(
        Eigen::Vector3d::Zero(), ray_id, timestamp));
      map_ptr_->AddParameterBlock(ray_errors_.front().back());

      Eigen::Vector3d target(cloud->at(i).x,
                             cloud->at(i).y,
                             cloud->at(i).z);
      std::shared_ptr<BodyVelocityCostFunction> doppler_cost_function =
        std::make_shared<BodyVelocityCostFunction>(cloud->at(i).doppler,
                                                   target,
                                                   weight,
                                                   d);
      // add residual block to ceres problem
      ceres::ResidualBlockId res_id =
					map_ptr_->AddResidualBlock(doppler_cost_function,
                                     doppler_loss_,
                                     velocities_.front(),
                                     ray_errors_.front().back());

      residual_blks_.front().push_back(res_id);
    }

    // add residual on change in velocity if applicable
    if (timestamps_.size() >= 2)
    {
      double delta_t = timestamps_[0] - timestamps_[1];
      std::shared_ptr<VelocityChangeCostFunction> vel_change_cost_func =
        std::make_shared<VelocityChangeCostFunction>(delta_t);

      ceres::ResidualBlockId res_id =
						map_ptr_->AddResidualBlock(vel_change_cost_func,
                                       accel_loss_,
                                       velocities_[1],
                                       velocities_[0]);

      residual_blks_[1].push_back(res_id);
    }
    // solve the ceres problem and get result
    map_ptr_->Solve();
    LOG(INFO) << map_ptr_->summary.FullReport();
    LOG(INFO) << "velocity from ceres: " << velocities_.front()->GetEstimate().transpose();

    // get estimate covariance for most recent speed only
    Eigen::MatrixXd covariance_matrix(3,3);
    covariance_matrix.setIdentity();
    /*
    std::vector<std::shared_ptr<ParameterBlock>> cov_params;
    cov_params.push_back(speeds_.front());
    map_ptr_->GetCovariance(cov_params,covariance_matrix);
    */

		covariance_matrix.diagonal() << 0.01, 0.015, 0.05;

    populateMessage(vel_out,velocities_.front()->GetEstimate(),covariance_matrix);
  }

  /** \brief populate ros message with velocity and covariance
    * \param[out] vel the resultant ros message
    * \param[in] velocity the estimated velocity
    * \param[in] covariance the estimate covariance
    */
  void populateMessage(geometry_msgs::TwistWithCovarianceStamped &vel,
                        const Eigen::Vector3d &velocity,
                        const Eigen::Matrix3d &covariance)
  {
		// get node namespace
    std::string ns = ros::this_node::getNamespace();

		if(ns.compare("/") == 0) {
    	// single radar frame_id to comply with TI naming convention
      vel.header.frame_id = "base_radar_link";
    }
    else {
      // multi-radar frame_id
    	vel.header.frame_id = ns.erase(0,1) + "_link";
    }

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
