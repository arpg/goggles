#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/impl/transforms.hpp>
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
#include <QuaternionParameterization.h>
#include <GlobalImuVelocityCostFunction.h>
#include <GlobalDopplerCostFunction.h>
#include <AHRSYawCostFunction.h>
#include <MarginalizationError.h>
#include "DataTypes.h"
#include "yaml-cpp/yaml.h"
#include <pcl/sample_consensus/impl/mlesac.hpp>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <BodyDopplerSacModel.h>
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


class RadarInertialVelocityReader
{
public:

  RadarInertialVelocityReader(ros::NodeHandle nh)
  {
    nh_ = nh;
		std::string radar_topic;
    std::string imu_topic;
		std::string imu_frame;
		std::string radar_frame;
		std::string config;
    nh_.getParam("radar_topic", radar_topic);
    nh_.getParam("imu_topic", imu_topic);
		nh_.getParam("imu_frame", imu_frame);
		nh_.getParam("radar_frame", radar_frame);
		nh_.getParam("config", config);
    nh_.getParam("publish_imu_state", publish_imu_propagated_state_);

		// get imu params and extrinsics
		LoadParams(config);
    imu_buffer_.SetTimeout(params_.frequency_);
		tf::TransformListener tf_listener;
		tf_listener.waitForTransform(imu_frame,
																 radar_frame,
																 ros::Time(0.0),
																 ros::Duration(1.0));
		tf_listener.lookupTransform(imu_frame,
																radar_frame,
																ros::Time(0.0),
																radar_to_imu_);
    Eigen::Quaterniond radar_to_imu_quat;
    tf::quaternionTFToEigen(radar_to_imu_.getRotation(),radar_to_imu_quat);
    radar_to_imu_mat_ = radar_to_imu_quat.toRotationMatrix();

		// get node namespace
    std::string ns = ros::this_node::getNamespace();

    pub_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(ns + "/mmWaveDataHdl/velocity",1);

		radar_sub_ = nh_.subscribe(radar_topic, 0, &RadarInertialVelocityReader::radarCallback, this);
		imu_sub_ = nh_.subscribe(imu_topic, 0, &RadarInertialVelocityReader::imuCallback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;
		initialized_ = false;
    first_state_optimized_ = false;

    window_size_ = 5;

    // set up ceres problem
    doppler_loss_ = new ceres::ScaledLoss(
      new ceres::CauchyLoss(1.0),1.0,ceres::DO_NOT_TAKE_OWNERSHIP);
    imu_loss_ = new ceres::ScaledLoss(
      new ceres::CauchyLoss(0.5),1.0,ceres::DO_NOT_TAKE_OWNERSHIP);
    yaw_loss_ = new ceres::ScaledLoss(
      NULL,50.0,ceres::DO_NOT_TAKE_OWNERSHIP);
		quat_param_ = new QuaternionParameterization;
    ceres::Problem::Options prob_options;

    prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    prob_options.enable_fast_removal = true;
    solver_options_.num_threads = 8;
    solver_options_.max_solver_time_in_seconds = 5.0e-2;
    solver_options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    problem_.reset(new ceres::Problem(prob_options));
  }

	void LoadParams(std::string config_filename)
	{
		YAML::Node config;
		try
		{
			config = YAML::LoadFile(config_filename);
		}
		catch (YAML::BadFile e)
		{
			LOG(FATAL) << "Bad config file at: " << config_filename << '\n' << e.msg;
		}

		// get imu params
		params_.frequency_ = config["frequency"].as<double>();
    params_.a_max_ = config["a_max"].as<double>();
    params_.g_max_ = config["g_max"].as<double>();
		params_.sigma_g_ = config["sigma_g"].as<double>();
		params_.sigma_a_ = config["sigma_a"].as<double>();
		params_.sigma_b_g_ = config["sigma_b_g"].as<double>();
		params_.sigma_b_a_ = config["sigma_b_a"].as<double>();
		params_.b_a_tau_ = config["b_a_tau"].as<double>();
    params_.invert_yaw_ = config["invert_yaw"].as<bool>();
    std::vector<double> vec = config["ahrs_to_imu"].as<std::vector<double>>();
    params_.ahrs_to_imu_ << vec[0], vec[1], vec[2],
                            vec[3], vec[4], vec[5],
                            vec[6], vec[7], vec[8];
	}

	void imuCallback(const sensor_msgs::ImuConstPtr& msg)
	{
		ImuMeasurement new_meas;

		new_meas.t_ = msg->header.stamp.toSec();

		new_meas.g_ << msg->angular_velocity.x,
									 msg->angular_velocity.y,
									 msg->angular_velocity.z;

		new_meas.a_ << msg->linear_acceleration.x,
									 msg->linear_acceleration.y,
									 msg->linear_acceleration.z;

    new_meas.q_.x() = msg->orientation.x;
    new_meas.q_.y() = msg->orientation.y;
    new_meas.q_.z() = msg->orientation.z;
    new_meas.q_.w() = msg->orientation.w;

    if (new_meas.g_.lpNorm<Eigen::Infinity>() > params_.g_max_)
      LOG(ERROR) << "Gyro saturation";
    if (new_meas.a_.lpNorm<Eigen::Infinity>() > params_.a_max_)
      LOG(ERROR) << "Accelerometer saturation";

		imu_buffer_.AddMeasurement(new_meas);

    if (publish_imu_propagated_state_ && first_state_optimized_)
    {
      //auto start = std::chrono::high_resolution_clock::now();
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
      propagateStateWithImu(new_meas.t_);
      //auto finish = std::chrono::high_resolution_clock::now();
      //std::chrono::duration<double> elapsed = finish - start;
      //LOG(ERROR) << "imu propagation time: " << elapsed.count();
      geometry_msgs::TwistWithCovarianceStamped vel_out;
      vel_out.header.stamp = msg->header.stamp;
      populateMessage(imu_propagated_speed_,vel_out,covariance);
      pub_.publish(vel_out);
    }
	}

  void radarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
	  double timestamp = msg->header.stamp.toSec();

		pcl::PointCloud<RadarPoint>::Ptr raw_cloud(new pcl::PointCloud<RadarPoint>);
	  pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
	  pcl::fromROSMsg(*msg, *raw_cloud);

		// undo stupid left-handed coordinate system from radar
		for (int i = 0; i < raw_cloud->size(); i++)
		{
			raw_cloud->at(i).y *= -1.0;
			raw_cloud->at(i).z *= -1.0;
		}

		// Reject clutter
    Declutter(raw_cloud);
		bool no_doppler = true;
		for (int i = 0; i < raw_cloud->size(); i++)
		{
			if (raw_cloud->at(i).doppler > 0)
				no_doppler = false;
		}
		// transform to imu frame
		pcl_ros::transformPointCloud(*raw_cloud, *cloud, radar_to_imu_);

		if (cloud->size() < 10)
			LOG(ERROR) << "input cloud has less than 10 points, output will be unreliable";

    // Get velocity measurements
    auto start = std::chrono::high_resolution_clock::now();
    GetVelocity(cloud, timestamp);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    sum_time_ += elapsed.count();
    num_iter_++;
    LOG(ERROR) << "execution time: " << sum_time_ / double(num_iter_);

    if (!publish_imu_propagated_state_)
    {
      geometry_msgs::TwistWithCovarianceStamped vel_out;
      vel_out.header.stamp = msg->header.stamp;
      Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
      populateMessage(*(speeds_.front()),vel_out,covariance_matrix);
      pub_.publish(vel_out);
    }
	}

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber radar_sub_;
	ros::Subscriber imu_sub_;
	tf::StampedTransform radar_to_imu_;
  Eigen::Matrix3d radar_to_imu_mat_;
  ceres::ScaledLoss *doppler_loss_;
  ceres::ScaledLoss *imu_loss_;
  ceres::ScaledLoss *yaw_loss_;
	ceres::LocalParameterization* quat_param_;
	std::deque<Eigen::Quaterniond*> orientations_;
  std::deque<Eigen::Vector3d*> speeds_;
  std::deque<Eigen::Vector3d*> gyro_biases_;
  std::deque<Eigen::Vector3d*> accel_biases_;
  Eigen::Matrix3d initial_orientation_;
  std::deque<double> timestamps_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
  std::shared_ptr<MarginalizationError> marginalization_error_ptr_;
  ceres::ResidualBlockId marginalization_id_;
	ImuBuffer imu_buffer_;
	ImuParams params_;
  std::shared_ptr<ceres::Problem> problem_;
  ceres::Solver::Options solver_options_;
  int window_size_;
  double min_range_;
  int num_iter_;
  double sum_time_;
	bool initialized_;
  bool publish_imu_propagated_state_;
  bool first_state_optimized_;
  std::mutex imu_propagation_mutex_;
  Eigen::Vector3d imu_propagated_speed_;
  Eigen::Quaterniond imu_propagated_orientation_;
  Eigen::Vector3d imu_propagated_g_bias_;
  Eigen::Vector3d imu_propagated_a_bias_;
  double propagated_state_timestamp_;
  double last_optimized_state_timestamp_;

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

  /** \brief uses ceres solver to estimate the global frame velocity from the input
    * point cloud
    * \param[in] raw_cloud the new pointcloud from the radar sensor
    * \param[in] timestamp the timestamp of the new point cloud
    */
  void GetVelocity(pcl::PointCloud<RadarPoint>::Ptr raw_cloud, double timestamp)
  {
		// if imu is not initialized, run initialization
		if (!initialized_)
		{
			InitializeImu();
		}
		// else add new params copied from previous values
		else
		{
			speeds_.push_front(new Eigen::Vector3d(*speeds_.front()));
      gyro_biases_.push_front(new Eigen::Vector3d(*gyro_biases_.front()));
      accel_biases_.push_front(new Eigen::Vector3d(*accel_biases_.front()));
			orientations_.push_front(new Eigen::Quaterniond(*orientations_.front()));
			problem_->AddParameterBlock(orientations_.front()->coeffs().data(),4);
			problem_->SetParameterization(orientations_.front()->coeffs().data(), quat_param_);
    	problem_->AddParameterBlock(speeds_.front()->data(),3);
			problem_->AddParameterBlock(gyro_biases_.front()->data(),3);
			problem_->AddParameterBlock(accel_biases_.front()->data(),3);
		}
    // add latest parameter blocks and remove old ones if necessary
    std::vector<ceres::ResidualBlockId> residuals;
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);

    // marginalize old states and measurements
    if (!ApplyMarginalization())
      LOG(ERROR) << "marginalization step failed";
    
    // use MLESAC to reject outlier measurements
    pcl::BodyDopplerSacModel<RadarPoint>::Ptr model(
      new pcl::BodyDopplerSacModel<RadarPoint>(raw_cloud));
    std::vector<int> inliers;
    pcl::MaximumLikelihoodSampleConsensus<RadarPoint> mlesac(model,0.1);
    mlesac.computeModel();
    mlesac.getInliers(inliers);

    // copy inlier points to new data structure;
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::copyPointCloud(*raw_cloud, inliers, *cloud);
    
    double weight = 1.0 / cloud->size();
    
		// add residuals on doppler readings
    for (int i = 0; i < cloud->size(); i++)
    {
			Eigen::Vector3d target(cloud->at(i).x,
                             cloud->at(i).y,
                             cloud->at(i).z);
      ceres::CostFunction* doppler_cost_function =
        new GlobalDopplerCostFunction(cloud->at(i).doppler,
                                      target,
                                      radar_to_imu_mat_,
                                      weight);

			// add residual block to ceres problem
      ceres::ResidualBlockId res_id =
				problem_->AddResidualBlock(doppler_cost_function,
                                   doppler_loss_,
                                   orientations_.front()->coeffs().data(),
                                	 speeds_.front()->data());
      residual_blks_.front().push_back(res_id);
      
    }

    // find imu measurements bracketing current timestep
    std::vector<ImuMeasurement> imu_measurements = 
      imu_buffer_.GetRange(timestamps_[0],timestamps_[0],false);
    if (imu_measurements.size() < 2)
      LOG(FATAL) << "not enough measurements returned";
    std::vector<ImuMeasurement>::iterator it = imu_measurements.begin();
    ImuMeasurement before, after;
    while ((it + 1) != imu_measurements.end())
    {
      if (it->t_ < timestamps_[0] && (it + 1)->t_ > timestamps_[0])
      {
        before = *it;
        after = *(it + 1);
      }
      it++;
    }
    if (before.t_ >= timestamps_[0] || after.t_ <= timestamps_[0])
      LOG(FATAL) << "imu measurements do not bracket current time";
      
    // use slerp to interpolate orientation at current timestep
    double r = (timestamps_[0] - before.t_) / (after.t_ - before.t_);
    Eigen::Quaterniond q_WS_t0 = before.q_.slerp(r, after.q_);

    // add constraint on yaw at current timestep
    ceres::CostFunction* yaw_cost_func = 
      new AHRSYawCostFunction(q_WS_t0,params_.invert_yaw_);

    ceres::ResidualBlockId orientation_res_id =
      problem_->AddResidualBlock(yaw_cost_func,
                                 yaw_loss_,
                                 orientations_.front()->coeffs().data());

    residual_blks_.front().push_back(orientation_res_id);
    
    // add imu cost only if there are more than 1 radar measurements in the queue
    if (timestamps_.size() >= 2)
    {
      //LOG(ERROR) << std::fixed << std::setprecision(3) << "adding imu cost from "
      //           << timestamps_[1] << " to " << timestamps_[0];
      bool delete_measurements = timestamps_[0] < propagated_state_timestamp_;
      //if (delete_measurements)
      //  LOG(ERROR) << "deleting measurements";
      //else
      //  LOG(ERROR) << "saving measurements";
      std::vector<ImuMeasurement> imu_measurements = 
					imu_buffer_.GetRange(timestamps_[1], timestamps_[0], delete_measurements);

			ceres::CostFunction* imu_cost_func =
        	new GlobalImuVelocityCostFunction(timestamps_[1],
																			      timestamps_[0],
																			      imu_measurements,
																			      params_);

			ceres::ResidualBlockId imu_res_id
				= problem_->AddResidualBlock(imu_cost_func,
                                     imu_loss_,
                                     orientations_[1]->coeffs().data(),
																		 speeds_[1]->data(),
																		 gyro_biases_[1]->data(),
																		 accel_biases_[1]->data(),
                                     orientations_[0]->coeffs().data(),
																		 speeds_[0]->data(),
																		 gyro_biases_[0]->data(),
																		 accel_biases_[0]->data());
			residual_blks_[1].push_back(imu_res_id);
    }
		// solve the ceres problem and get result
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);

    if (publish_imu_propagated_state_)
    {
      std::lock_guard<std::mutex> lck(imu_propagation_mutex_);
      propagated_state_timestamp_ = timestamps_[0];
      last_optimized_state_timestamp_ = timestamps_[0];
      imu_propagated_orientation_ = *(orientations_.front());
      imu_propagated_speed_ = *(speeds_.front());
      imu_propagated_g_bias_ = *(gyro_biases_.front());
      imu_propagated_a_bias_ = *(accel_biases_.front());
      first_state_optimized_ = true;
    }
    
    /*
    LOG(INFO) << summary.FullReport();
    std::ofstream orientation_log;
    std::string filename = "/home/akramer/logs/radar/ICRA_2020/orientations.csv";
    orientation_log.open(filename,std::ofstream::app);
    orientation_log << std::fixed << std::setprecision(5) << timestamps_.front()
                    << ',' << orientations_.front()->x() << ',' << orientations_.front()->y()
                    << ',' << orientations_.front()->z() << ',' << orientations_.front()->w()
                    << '\n';
    orientation_log.close();
    */
    // get estimate covariance
    /*
    ceres::Covariance::Options cov_options;
    cov_options.num_threads = 1;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(cov_options);

    std::vector<std::pair<const double*, const double*>> cov_blks;
    cov_blks.push_back(std::make_pair(speeds_and_biases_.front()->head<3>().data(),
                                      speeds_and_biases_.front()->head<3>().data()));

    covariance.Compute(cov_blks, problem_.get());
    Eigen::Matrix3d covariance_matrix;
    covariance.GetCovarianceBlock(speeds_and_biases_.front()->head<3>().data(),
                                  speeds_and_biases_.front()->head<3>().data(),
                                  covariance_matrix.data());

    LOG(INFO) << "covariance: \n" << covariance_matrix;
    */
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
    
  }

  /** \brief uses initial IMU measurements to determine the magnitude of the
    * gravity vector and the initial attitude
    */
  void InitializeImu()
  {
    LOG(ERROR) << "initializing state from imu";
    imu_buffer_.WaitForMeasurements();
    double t0 = imu_buffer_.GetStartTime();
    std::vector<ImuMeasurement> measurements =
          imu_buffer_.GetRange(t0, t0 + 0.2, false);

    // find gravity vector and average stationary gyro reading
    Eigen::Vector3d sum_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < measurements.size(); i++)
    {
      sum_g += measurements[i].g_;
      sum_a += measurements[i].a_;
    }
    Eigen::Vector3d g_vec = sum_a / double(measurements.size());

    // set gravity vector magnitude
    params_.g_ = -g_vec.norm();

    // set initial velocity and biases
    Eigen::Vector3d speed_initial = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_g_initial = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_a_initial = Eigen::Vector3d::Zero();
    b_g_initial = sum_g / measurements.size();
    speeds_.push_front(new Eigen::Vector3d(speed_initial));
    gyro_biases_.push_front(new Eigen::Vector3d(b_g_initial));
    accel_biases_.push_front(new Eigen::Vector3d(b_a_initial));

    // set initial orientation
    // assuming IMU is set close to Z-down
    Eigen::Vector3d down_vec(0.0,0.0,1.0);
    Eigen::Vector3d increment;
    Eigen::Vector3d g_direction = -g_vec.normalized();
    Eigen::Vector3d cross = down_vec.cross(g_direction);
    if (cross.norm() == 0.0)
      increment = cross;
    else
      increment = cross.normalized();
    double angle = std::acos(down_vec.transpose() * g_direction);
    increment *= angle;
    increment *= -1.0;

    LOG(ERROR) << "      angle: " << angle;
    LOG(ERROR) << "g_direction: " << g_direction.transpose();

    Eigen::Quaterniond initial_orientation = Eigen::Quaterniond::Identity();
    // initial oplus increment
    QuaternionParameterization qp;
    Eigen::Quaterniond dq = qp.DeltaQ(increment);

    initial_orientation = (dq * initial_orientation);
    initial_orientation.normalize();

    orientations_.push_front(
        new Eigen::Quaterniond(initial_orientation));
    initial_orientation_ = (params_.ahrs_to_imu_ * 
      measurements.front().q_.toRotationMatrix()).inverse() * 
      initial_orientation.toRotationMatrix();

    problem_->AddParameterBlock(orientations_.front()->coeffs().data(),4);
    problem_->SetParameterization(orientations_.front()->coeffs().data(), quat_param_);
    problem_->AddParameterBlock(speeds_.front()->data(),3);
    problem_->AddParameterBlock(gyro_biases_.front()->data(),3);
    problem_->AddParameterBlock(accel_biases_.front()->data(),3);
    problem_->SetParameterBlockConstant(gyro_biases_.front()->data());
    problem_->SetParameterBlockConstant(accel_biases_.front()->data());

    initialized_ = true;
    LOG(ERROR) << "initialized!";
    LOG(ERROR) << "initial orientation: " << initial_orientation.coeffs().transpose();
    LOG(ERROR) << "initial biases: " << b_g_initial.transpose();
  }

  /** \brief linearize old states and measurements and add them to the
    * marginalization error term
    */
  bool ApplyMarginalization()
  {
    if (timestamps_.size() > window_size_)
    {
      // remove marginalization error from problem
      // if it's already initialized
      if (marginalization_error_ptr_ && marginalization_id_)
      {
        problem_->RemoveResidualBlock(marginalization_id_);
        marginalization_id_ = 0;
      }

      // initialize the marginalization error if necessary
      if (!marginalization_error_ptr_)
      {
        marginalization_error_ptr_.reset(
          new MarginalizationError(problem_));
      }

      // add oldest residuals
      if(!marginalization_error_ptr_->AddResidualBlocks(residual_blks_.back()))
      {
        LOG(ERROR) << "failed to add residuals";
        return false;
      }

      // get oldest states to marginalize
      std::vector<double*> states_to_marginalize;
      states_to_marginalize.push_back(
        orientations_.back()->coeffs().data()); // attitude
      states_to_marginalize.push_back(
        speeds_.back()->data()); // speed
      states_to_marginalize.push_back(
        gyro_biases_.back()->data()); // gyro bias
      states_to_marginalize.push_back(
        accel_biases_.back()->data()); // accel bias

      // actually marginalize states
      if (!marginalization_error_ptr_->MarginalizeOut(states_to_marginalize))
      {
        LOG(ERROR) << "failed to marginalize states";
        return false;
      }

      marginalization_error_ptr_->UpdateErrorComputation();

      orientations_.pop_back();
      speeds_.pop_back();
      gyro_biases_.pop_back();
      accel_biases_.pop_back();
      residual_blks_.pop_back();
      timestamps_.pop_back();

      // discard marginalization error if it has no residuals
      if (marginalization_error_ptr_->ResidualDim() == 0)
      {
        LOG(ERROR) << "no residuals associated to marginalization, resetting";
        marginalization_error_ptr_.reset();
      }

      // add marginalization error term back to the problem
      if (marginalization_error_ptr_)
      {
        std::vector<double*> parameter_block_ptrs;
        marginalization_error_ptr_->GetParameterBlockPtrs(
          parameter_block_ptrs);
        marginalization_id_ = problem_->AddResidualBlock(
          marginalization_error_ptr_.get(),
          NULL,
          parameter_block_ptrs);
      }
    }
    return true;
  }

  /** \brief propagates state from last optimized state to time of most recent
    * imu measurement
    * \param[in] timestamp of the most recent imu measurement
    */
  void propagateStateWithImu(double t1)
  {
    // ensure unique access
    std::lock_guard<std::mutex> lck(imu_propagation_mutex_);
    //LOG(ERROR) << std::fixed << std::setprecision(5) 
    //           << "propagating imu state from " << propagated_state_timestamp_
    //           << " to " << t1;

    // propagate state using imu measurement
    std::vector<ImuMeasurement> imu_measurements;
    bool delete_measurements = propagated_state_timestamp_ < last_optimized_state_timestamp_;
    imu_measurements = imu_buffer_.GetRange(propagated_state_timestamp_, t1, delete_measurements);
    //LOG(ERROR) << std::fixed << std::setprecision(5) 
    //           << "doing propagation with measurements from "
    //           << imu_measurements.front().t_ << " to " 
    //           << imu_measurements.back().t_;
    GlobalImuVelocityCostFunction::Propagation(imu_measurements,
                                               params_,
                                               imu_propagated_orientation_,
                                               imu_propagated_speed_,
                                               imu_propagated_g_bias_,
                                               imu_propagated_a_bias_,
                                               propagated_state_timestamp_,
                                               t1);

    propagated_state_timestamp_ = t1;
    //LOG(ERROR) << "done propagating state";
    //LOG(ERROR) << "imu buffer size: " << imu_buffer_.size();
  }

  /** \brief populate ros message with velocity and covariance
    * \param[in] velocity the estimated velocity
    * \param[out] vel the resultant ros message
    * \param[in] covariance the estimate covariance
    */
  void populateMessage(Eigen::Vector3d velocity,
                       geometry_msgs::TwistWithCovarianceStamped &vel,
                       Eigen::Matrix3d &covariance)
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

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
        vel.twist.covariance[(i*6)+j] = covariance(i,j);
    }
  }

};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
	ros::init(argc, argv, "radar_vel");
  ros::NodeHandle nh("~");
  RadarInertialVelocityReader* rv_reader = new RadarInertialVelocityReader(nh);

  //ros::spin();
	ros::AsyncSpinner spinner(4);
	spinner.start();
	ros::waitForShutdown();

  return 0;
}
