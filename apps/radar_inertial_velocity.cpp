#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
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
#include <OrientationParameterBlock.h>
#include <VelocityParameterBlock.h>
#include <BiasParameterBlock.h>
#include <DeltaParameterBlock.h>
#include <Map.h>
#include <IdProvider.h>
#include "DataTypes.h"
#include "yaml-cpp/yaml.h"
#include <pcl/sample_consensus/impl/mlesac.hpp>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <BodyDopplerSacModel.h>
#include <chrono>

class RadarInertialVelocityReader
{
public:

  RadarInertialVelocityReader(ros::NodeHandle nh) : map_ptr_(new Map())
  {
    nh_ = nh;
		int num_radar = 1;
		std::string config;
    nh_.param("num_radar", num_radar, num_radar); // number of radar boards used
		nh_.getParam("config", config);
    nh_.getParam("publish_imu_state", publish_imu_propagated_state_);
    nh_.getParam("publish_inliers", publish_inliers_);
    imu_frame_ = "";

		// get imu params
		LoadParams(config);
    imu_buffer_.SetTimeout(params_.frequency_);
		
    ros::Duration(0.5).sleep();

    odom_frame_id_ = "radar_odom_frame";

		// get node namespace
    std::string ns = ros::this_node::getNamespace();

    velocity_publisher_ = nh_.advertise<nav_msgs::Odometry>(
      ns + "/mmWaveDataHdl/velocity",1);
    if (publish_inliers_) 
      inlier_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
        ns + "/mmWaveDataHdl/inlier_set",1);

    // split radar subscriber list and create subscribers
    std::string base_radar_topic = "/radar_";
    for (int i = 0; i < num_radar; i++)
    {
      std::string radar_topic = base_radar_topic + std::to_string(i);
      radar_subs_.push_back(
        nh_.subscribe(radar_topic,
                      1,
                      &RadarInertialVelocityReader::radarCallback,
                      this));
    }

		imu_sub_ = nh_.subscribe("/imu_data", 
                             1, 
                             &RadarInertialVelocityReader::imuCallback, 
                             this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;
		initialized_ = false;
    first_state_optimized_ = false;

    window_size_ = 5;

    // set up ceres problem
    doppler_loss_ = new ceres::CauchyLoss(1.0);
    imu_loss_ = new ceres::CauchyLoss(1.0);
    yaw_loss_ = new ceres::ScaledLoss(
      NULL,50.0,ceres::DO_NOT_TAKE_OWNERSHIP);
		quat_param_ = new QuaternionParameterization;
    map_ptr_->options.num_threads = 8;
    //map_ptr_->options.max_solver_time_in_seconds = 5.0e-2;
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
    imu_frame_ = msg->header.frame_id;

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
    //LOG(ERROR) << "got imu with ts: " << std::fixed << std::setprecision(4) << new_meas.t_;
    if (new_meas.g_.lpNorm<Eigen::Infinity>() > params_.g_max_)
      LOG(ERROR) << "Gyro saturation";
    if (new_meas.a_.lpNorm<Eigen::Infinity>() > params_.a_max_)
      LOG(ERROR) << "Accelerometer saturation";

		imu_buffer_.AddMeasurement(new_meas);

    if (publish_imu_propagated_state_ && first_state_optimized_)
    {
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
      propagateStateWithImu(new_meas.t_);
      nav_msgs::Odometry odom_out;
      odom_out.header.stamp = msg->header.stamp;
      populateMessage(imu_propagated_speed_,
                      imu_propagated_orientation_,
                      odom_out,
                      covariance);
      velocity_publisher_.publish(odom_out);
    }
	}

  void radarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
	  if (imu_frame_ != "")
    {
      double timestamp = msg->header.stamp.toSec();

  		pcl::PointCloud<RadarPoint>::Ptr raw_cloud(new pcl::PointCloud<RadarPoint>);
  	  pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
  	  pcl::fromROSMsg(*msg, *raw_cloud);

  		// Reject clutter
      Declutter(raw_cloud);
  		bool no_doppler = true;
  		for (int i = 0; i < raw_cloud->size(); i++)
  		{
  			if (raw_cloud->at(i).doppler > 0)
  				no_doppler = false;
  		}
      std::string radar_frame = msg->header.frame_id;

      // get radar to imu transform
      tf::StampedTransform radar_to_imu;
      tf_listener_.lookupTransform(imu_frame_,
                                   radar_frame,
                                   ros::Time(0.0),
                                   radar_to_imu);
      Eigen::Quaterniond radar_to_imu_quat;
      tf::quaternionTFToEigen(radar_to_imu.getRotation(),radar_to_imu_quat);
      Eigen::Matrix3d radar_to_imu_mat = radar_to_imu_quat.toRotationMatrix();

  		// transform to imu frame
  		pcl_ros::transformPointCloud(*raw_cloud, *cloud, radar_to_imu);

  		if (cloud->size() < 10)
  			LOG(ERROR) << "input cloud has less than 10 points, output will be unreliable";

      // Get velocity measurements
      auto start = std::chrono::high_resolution_clock::now();
      GetVelocity(cloud, timestamp, radar_to_imu_mat);
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      sum_time_ += elapsed.count();
      num_iter_++;
      LOG(INFO) << "execution time: " << sum_time_ / double(num_iter_);

      if (!publish_imu_propagated_state_)
      {
        nav_msgs::Odometry odom_out;
        odom_out.header.stamp = msg->header.stamp;
        Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
        populateMessage(speeds_.front()->GetEstimate(),
                        orientations_.front()->GetEstimate(),
                        odom_out,
                        covariance_matrix);
        velocity_publisher_.publish(odom_out);
      }
    }
	}

private:
  // ros-related objects
  ros::NodeHandle nh_;
  ros::Publisher velocity_publisher_;
  ros::Publisher inlier_publisher_;
  std::vector<ros::Subscriber> radar_subs_;
	ros::Subscriber imu_sub_;
  // ceres objects
  std::shared_ptr<Map> map_ptr_;
  ceres::LossFunction *doppler_loss_;
  ceres::LossFunction *imu_loss_;
  ceres::LossFunction *yaw_loss_;
  ceres::ResidualBlockId marginalization_id_;

  // optimized state and residual containers
	QuaternionParameterization* quat_param_;
	std::deque<std::shared_ptr<OrientationParameterBlock>> orientations_;
  std::deque<std::shared_ptr<VelocityParameterBlock>> speeds_;
  std::deque<std::shared_ptr<BiasParameterBlock>> gyro_biases_;
  std::deque<std::shared_ptr<BiasParameterBlock>> accel_biases_;
  std::deque<std::vector<std::shared_ptr<DeltaParameterBlock>>> ray_errors_;
  Eigen::Matrix3d initial_orientation_;
  Eigen::Quaterniond initial_yaw_;
  std::deque<double> timestamps_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
  std::shared_ptr<MarginalizationError> marginalization_error_ptr_;
  bool initialized_;
  
  // imu sensor buffer and parameters
	ImuBuffer imu_buffer_;
	ImuParams params_;
	
  // imu propagated state containers
  bool publish_imu_propagated_state_;
  bool first_state_optimized_;
  double propagated_state_timestamp_;
  double last_optimized_state_timestamp_;
  std::mutex imu_propagation_mutex_;
  std::mutex optimization_mutex_;
  Eigen::Vector3d imu_propagated_speed_;
  Eigen::Quaterniond imu_propagated_orientation_;
  Eigen::Vector3d imu_propagated_g_bias_;
  Eigen::Vector3d imu_propagated_a_bias_;

  int window_size_;
  double min_range_;
  int num_iter_;
  double sum_time_;
  bool publish_inliers_;
  std::string odom_frame_id_;
  std::string imu_frame_;
  tf::TransformListener tf_listener_;


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
  void GetVelocity(pcl::PointCloud<RadarPoint>::Ptr raw_cloud, 
                   double timestamp,
                   Eigen::Matrix3d &radar_to_imu_mat)
  {
    std::lock_guard<std::mutex> optimization_lock(optimization_mutex_);
		// if imu is not initialized, run initialization
		if (!initialized_)
		{
			if (!InitializeImu(timestamp))
        return;
		}
		// else add new params copied from previous values
		else
		{
      uint64_t id = IdProvider::instance().NewId();
  		speeds_.push_front(std::make_shared<VelocityParameterBlock>(
        speeds_.front()->GetEstimate(),id,timestamp));

      id = IdProvider::instance().NewId();
      gyro_biases_.push_front(std::make_shared<BiasParameterBlock>(
        gyro_biases_.front()->GetEstimate(),id,timestamp));

      id = IdProvider::instance().NewId();
      accel_biases_.push_front(std::make_shared<BiasParameterBlock>(
        accel_biases_.front()->GetEstimate(),id,timestamp));

      id = IdProvider::instance().NewId();
  		orientations_.push_front(std::make_shared<OrientationParameterBlock>(
        orientations_.front()->GetEstimate(),id,timestamp));

			map_ptr_->AddParameterBlock(orientations_.front(),
        Map::Parameterization::Orientation);
    	map_ptr_->AddParameterBlock(speeds_.front());
			map_ptr_->AddParameterBlock(gyro_biases_.front());
			map_ptr_->AddParameterBlock(accel_biases_.front());
		}

    // add latest parameter blocks and remove old ones if necessary
    std::vector<ceres::ResidualBlockId> residuals;
    std::vector<std::shared_ptr<DeltaParameterBlock>> ray_err;
    ray_errors_.push_front(ray_err);
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);

    // marginalize old states and measurements
    if (!ApplyMarginalization())
      LOG(ERROR) << "marginalization step failed";
    
    // use MLESAC to reject outlier measurements
    pcl::BodyDopplerSacModel<RadarPoint>::Ptr model(
      new pcl::BodyDopplerSacModel<RadarPoint>(raw_cloud));
    std::vector<int> inliers;
    pcl::MaximumLikelihoodSampleConsensus<RadarPoint> mlesac(model,0.15);
    mlesac.computeModel();
    mlesac.getInliers(inliers);
    
    // copy inlier points to new data structure;
    pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
    pcl::copyPointCloud(*raw_cloud, inliers, *cloud);    

    // publish inlier set, if requested
    if (publish_inliers_) PublishInliers(cloud, timestamp);

    double weight = 2.5 / cloud->size();
    double d = 0.95;
    
    // add residuals on doppler readings
    for (int i = 0; i < cloud->size(); i++)
    {
      Eigen::Vector3d target(cloud->at(i).x,
         cloud->at(i).y,
         cloud->at(i).z);
      std::shared_ptr<ceres::CostFunction> doppler_cost_function =
      std::make_shared<GlobalDopplerCostFunction>(cloud->at(i).doppler,
        target,
        radar_to_imu_mat,
        weight,
        d);
      uint64_t id = IdProvider::instance().NewId();
      ray_errors_.front().push_back(std::make_shared<DeltaParameterBlock>(
        Eigen::Vector3d::Zero(), id, timestamp));

      map_ptr_->AddParameterBlock(ray_errors_.front().back());

    	// add residual block to ceres problem
      ceres::ResidualBlockId res_id =
      map_ptr_->AddResidualBlock(doppler_cost_function,
        doppler_loss_,
        orientations_.front(),
        speeds_.front(),
        ray_errors_.front().back());
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
      if (it->t_ <= timestamps_[0] && (it + 1)->t_ >= timestamps_[0])
      {
        before = *it;
        after = *(it + 1);
      }
      it++;
    }
    if (before.t_ > timestamps_[0] || after.t_ < timestamps_[0])
      LOG(FATAL) << "imu measurements do not bracket current time";

    // use slerp to interpolate orientation at current timestep
    double r = (timestamps_[0] - before.t_) / (after.t_ - before.t_);
    Eigen::Quaterniond q_WS_t0 = before.q_.slerp(r, after.q_);
    q_WS_t0 = initial_yaw_.conjugate() * q_WS_t0;

    // add constraint on yaw at current timestep
    std::shared_ptr<ceres::CostFunction> yaw_cost_func = 
    std::make_shared<AHRSYawCostFunction>(q_WS_t0,params_.invert_yaw_);

    ceres::ResidualBlockId orientation_res_id =
    map_ptr_->AddResidualBlock(yaw_cost_func,
     yaw_loss_,
     orientations_.front());

    residual_blks_.front().push_back(orientation_res_id);
    
    // add imu cost only if there are more than 1 radar measurements in the queue
    if (timestamps_.size() >= 2)
    {
      for (size_t i = 0; i < ray_errors_[1].size(); i++)
        map_ptr_->SetParameterBlockConstant(ray_errors_[1][i]);

      bool delete_measurements = timestamps_[0] < propagated_state_timestamp_;

      std::vector<ImuMeasurement> imu_measurements = 
      imu_buffer_.GetRange(timestamps_[1], 
       timestamps_[0], 
       delete_measurements);

      std::shared_ptr<ceres::CostFunction> imu_cost_func =
      std::make_shared<GlobalImuVelocityCostFunction>(timestamps_[1],
        timestamps_[0],
        imu_measurements,
        params_);

      ceres::ResidualBlockId imu_res_id = map_ptr_->AddResidualBlock(imu_cost_func,
        imu_loss_,
        orientations_[1],
        speeds_[1],
        gyro_biases_[1],
        accel_biases_[1],
        orientations_[0],
        speeds_[0],
        gyro_biases_[0],
        accel_biases_[0]);
      residual_blks_[1].push_back(imu_res_id);
    }
    
  	// solve the ceres problem and get result
    map_ptr_->Solve();
    LOG(INFO) << map_ptr_->summary.FullReport();

    if (publish_imu_propagated_state_)
    {
      std::lock_guard<std::mutex> lck(imu_propagation_mutex_);
      propagated_state_timestamp_ = timestamps_[0];
      last_optimized_state_timestamp_ = timestamps_[0];
      imu_propagated_orientation_ = orientations_.front()->GetEstimate();
      imu_propagated_speed_ = speeds_.front()->GetEstimate();
      imu_propagated_g_bias_ = gyro_biases_.front()->GetEstimate();
      imu_propagated_a_bias_ = accel_biases_.front()->GetEstimate();
      first_state_optimized_ = true;
    }
    
    /*
    std::ofstream orientation_log;
    std::string filename = "/home/akramer/logs/radar/ICRA_2020/orientations.csv";
    orientation_log.open(filename,std::ofstream::app);
    orientation_log << std::fixed << std::setprecision(5) << timestamps_.front()
                    << ',' << orientations_.front()->x() << ',' << orientations_.front()->y()
                    << ',' << orientations_.front()->z() << ',' << orientations_.front()->w()
                    << '\n';
    orientation_log.close();
    */
    // get estimate covariance for most recent speed only
    Eigen::MatrixXd covariance_matrix(3,3);
    covariance_matrix.setIdentity();
    /*
    std::vector<std::shared_ptr<ParameterBlock>> cov_params;
    cov_params.push_back(speeds_.front());
    map_ptr_->GetCovariance(cov_params,covariance_matrix);
    */
    LOG(INFO) << "covariance: \n" << covariance_matrix;
    
  }

  /** \brief uses initial IMU measurements to determine the magnitude of the
    * gravity vector and the initial attitude
    */
  bool InitializeImu(double timestamp)
  {
    imu_buffer_.WaitForMeasurements();

    double start_t = imu_buffer_.GetStartTime();
    double t1 = imu_buffer_.GetEndTime();
    double duration = 0.5;
    if (t1 - start_t < duration)
	    return false;

    std::vector<ImuMeasurement> measurements =
          imu_buffer_.GetRange(t1 - duration, t1, false);

    // some IMUs give erratic readings when they're first starting
    // check for these and throw them out if found
    double delete_end_time = measurements.front().t_;
    double imu_period = 1.0/params_.frequency_;
    int num_to_delete = 0;
    for (int i = 1; i < measurements.size(); i++)
    {
      if (std::fabs((measurements[i].t_ - measurements[i-1].t_) - imu_period) > imu_period)
      {
        delete_end_time = measurements[i].t_;
        num_to_delete++;
      }  
    }
    if (num_to_delete > 0)
    {
      imu_buffer_.GetRange(start_t, delete_end_time, true);
      return false;
    }

    if (measurements.size() < duration * params_.frequency_)
		  return false;
    
    LOG(ERROR) << "initializing states from imu measurements"; 
	
    // find gravity vector and average stationary gyro reading
    Eigen::Vector3d sum_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < measurements.size(); i++)
    {
      sum_g += measurements[i].g_;
      sum_a += measurements[i].a_;
    }
    Eigen::Vector3d g_vec = sum_a / double(measurements.size());
    initial_yaw_ = Eigen::Quaterniond(measurements.back().q_.w(),
                                      measurements.back().q_.x(),
                                      measurements.back().q_.y(),
                                      measurements.back().q_.z());

    // set gravity vector magnitude
    params_.g_ = g_vec.norm();

    // set initial velocity and biases
    Eigen::Vector3d speed_initial = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_g_initial = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_a_initial = Eigen::Vector3d::Zero();
    b_g_initial = sum_g / measurements.size();

    uint64_t id = IdProvider::instance().NewId();
    speeds_.push_front(std::make_shared<VelocityParameterBlock>(
      speed_initial,id,timestamp));

    id = IdProvider::instance().NewId();
    gyro_biases_.push_front(std::make_shared<BiasParameterBlock>(
      b_g_initial,id,timestamp));

    id = IdProvider::instance().NewId();
    accel_biases_.push_front(std::make_shared<BiasParameterBlock>(
      b_a_initial,id,timestamp));

    // set initial orientation
    // assuming IMU is set close to Z-down
    Eigen::Vector3d down_vec(0.0,0.0,1.0);
    Eigen::Vector3d increment;
    Eigen::Vector3d g_direction = g_vec.normalized();
    Eigen::Vector3d cross = down_vec.cross(g_direction);
    if (cross.norm() == 0.0)
      increment = cross;
    else
      increment = cross.normalized();
    double angle = std::acos(down_vec.transpose() * g_direction);
    increment *= angle;
    increment *= -1.0;

    LOG(ERROR) << "g magnitude: " << params_.g_;
    LOG(ERROR) << "      angle: " << angle;
    LOG(ERROR) << "g_direction: " << g_direction.transpose();

    Eigen::Quaterniond initial_orientation = Eigen::Quaterniond::Identity();
    // initial oplus increment
    QuaternionParameterization qp;
    Eigen::Quaterniond dq = qp.DeltaQ(increment);

    initial_orientation = (dq * initial_orientation);
    initial_orientation.normalize();

    id = IdProvider::instance().NewId();
    orientations_.push_front(std::make_shared<OrientationParameterBlock>(
      initial_orientation,id,timestamp));
    initial_orientation_ = (params_.ahrs_to_imu_ * 
      measurements.front().q_.toRotationMatrix()).inverse() * 
      initial_orientation.toRotationMatrix();

    map_ptr_->AddParameterBlock(orientations_.front(),
      Map::Parameterization::Orientation);
    map_ptr_->AddParameterBlock(speeds_.front());
    map_ptr_->AddParameterBlock(gyro_biases_.front());
    map_ptr_->AddParameterBlock(accel_biases_.front());
    map_ptr_->SetParameterBlockConstant(gyro_biases_.front());
    map_ptr_->SetParameterBlockConstant(accel_biases_.front());

    initialized_ = true;
    LOG(ERROR) << "initialized!";
    LOG(ERROR) << "initial orientation: " << initial_orientation.coeffs().transpose();
    LOG(ERROR) << "initial biases: " << b_g_initial.transpose();

    return true;
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
        map_ptr_->RemoveResidualBlock(marginalization_id_);
        marginalization_id_ = 0;
      }

      // initialize the marginalization error if necessary
      if (!marginalization_error_ptr_)
      {
        marginalization_error_ptr_.reset(
          new MarginalizationError(map_ptr_));
      }

      // add oldest residuals
      if(!marginalization_error_ptr_->AddResidualBlocks(residual_blks_.back()))
      {
        LOG(ERROR) << "failed to add residuals";
        return false;
      }

      // get oldest states to marginalize
      std::vector<uint64_t> states_to_marginalize;
      states_to_marginalize.push_back(
        orientations_.back()->GetId()); // attitude
      states_to_marginalize.push_back(
        speeds_.back()->GetId()); // speed
      states_to_marginalize.push_back(
        gyro_biases_.back()->GetId()); // gyro bias
      states_to_marginalize.push_back(
        accel_biases_.back()->GetId()); // accel bias
      for (int i = 0; i < ray_errors_.back().size(); i++)
        states_to_marginalize.push_back(ray_errors_.back()[i]->GetId());

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
      ray_errors_.pop_back();

      // discard marginalization error if it has no residuals
      if (marginalization_error_ptr_->ResidualDim() == 0)
      {
        LOG(ERROR) << "no residuals associated to marginalization, resetting";
        marginalization_error_ptr_.reset();
      }

      // add marginalization error term back to the problem
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

    // propagate state using imu measurement
    std::vector<ImuMeasurement> imu_measurements;
    bool delete_measurements = 
      propagated_state_timestamp_ < last_optimized_state_timestamp_;
    imu_measurements = imu_buffer_.GetRange(
      propagated_state_timestamp_, t1, delete_measurements);

    GlobalImuVelocityCostFunction::Propagation(imu_measurements,
                                               params_,
                                               imu_propagated_orientation_,
                                               imu_propagated_speed_,
                                               imu_propagated_g_bias_,
                                               imu_propagated_a_bias_,
                                               propagated_state_timestamp_,
                                               t1);

    propagated_state_timestamp_ = t1;
  }

  /** \brief publishes inlier points after MLESAC step
    * \param[in] cloud The inlier point cloud
    * \param[in] timestamp of the outgoing point cloud
    */
  void PublishInliers(pcl::PointCloud<RadarPoint>::Ptr imu_frame_cloud, double timestamp)
  {
    // transform point cloud back to the radar frame
    pcl::PointCloud<RadarPoint>::Ptr radar_frame_cloud(new pcl::PointCloud<RadarPoint>());
    //pcl_ros::transformPointCloud(*imu_frame_cloud, *radar_frame_cloud, imu_to_radar_);

    // redo stupid left-handed coordinate system from radar
    /*
    for (int i = 0; i < radar_frame_cloud->size(); i++)
    {
      radar_frame_cloud->at(i).y *= -1.0;
      radar_frame_cloud->at(i).z *= -1.0;
    }
    */
    // convert to PCL2 type
    pcl::PCLPointCloud2 radar_frame_cloud2;
    pcl::toPCLPointCloud2(*imu_frame_cloud, radar_frame_cloud2);

    // convert to ros message type
    sensor_msgs::PointCloud2 out_cloud;
    out_cloud.header.stamp = ros::Time(timestamp);
    
    pcl_conversions::fromPCL(radar_frame_cloud2, out_cloud);
    out_cloud.header.frame_id = imu_frame_;
    inlier_publisher_.publish(out_cloud);
  }

  /** \brief populate ros message with velocity and covariance
    * \param[in] velocity the estimated velocity
    * \param[out] vel the resultant ros message
    * \param[in] covariance the estimate covariance
    */
  void populateMessage(Eigen::Vector3d velocity,
                       Eigen::Quaterniond orientation,
                       nav_msgs::Odometry &odom,
                       Eigen::Matrix3d &covariance)
  {
   	// set frame id
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = imu_frame_;

    Eigen::Vector3d v_s = orientation.toRotationMatrix().inverse() * velocity;
    odom.twist.twist.linear.x = v_s.x();
    odom.twist.twist.linear.y = v_s.y();
    odom.twist.twist.linear.z = v_s.z();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.pose.pose.orientation.w = orientation.w();

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
        odom.twist.covariance[(i*6)+j] = covariance(i,j);
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
