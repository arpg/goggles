#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <tf/transform_listener.h>
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
#include <MarginalizationError.h>
#include "DataTypes.h"
#include "yaml-cpp/yaml.h"
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

		// get node namespace
    std::string ns = ros::this_node::getNamespace();

    pub_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(ns + "/mmWaveDataHdl/velocity",1);

		radar_sub_ = nh_.subscribe(radar_topic, 0, &RadarInertialVelocityReader::radarCallback, this);
		imu_sub_ = nh_.subscribe(imu_topic, 0, &RadarInertialVelocityReader::imuCallback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;
		initialized_ = false;

    window_size_ = 3;

    // set up ceres problem
    doppler_loss_ = new ceres::CauchyLoss(.15);
    imu_loss_ = new ceres::ScaledLoss(new ceres::CauchyLoss(1.0),1.0,ceres::DO_NOT_TAKE_OWNERSHIP);
		quat_param_ = new QuaternionParameterization;
    ceres::Problem::Options prob_options;

    prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    prob_options.enable_fast_removal = true;
    //solver_options_.check_gradients = true;
    //solver_options_.gradient_check_relative_precision = 1.0e-4;
    solver_options_.num_threads = 8;
    solver_options_.max_num_iterations = 300;
    solver_options_.update_state_every_iteration = true;
    solver_options_.function_tolerance = 1e-10;
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

    if (new_meas.g_.lpNorm<Eigen::Infinity>() > params_.g_max_)
      LOG(ERROR) << "Gyro saturation";
    if (new_meas.a_.lpNorm<Eigen::Infinity>() > params_.a_max_)
      LOG(ERROR) << "Accelerometer saturation";

		imu_buffer_.AddMeasurement(new_meas);
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
		if (no_doppler)
		{
			LOG(ERROR) << std::fixed << std::setprecision(5) << "no doppler reading at " << timestamp;
			return;
		}
		// transform to imu frame
		pcl_ros::transformPointCloud(*raw_cloud, *cloud, radar_to_imu_);

		if (cloud->size() < 10)
			LOG(ERROR) << "input cloud has less than 10 points, output will be unreliable";

    geometry_msgs::TwistWithCovarianceStamped vel_out;
    vel_out.header.stamp = msg->header.stamp;

    // Get velocity measurements
    auto start = std::chrono::high_resolution_clock::now();
    GetVelocity(cloud, timestamp, vel_out);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    sum_time_ += elapsed.count();
    num_iter_++;
    LOG(ERROR) << "execution time: " << sum_time_ / double(num_iter_);
    pub_.publish(vel_out);

	}

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber radar_sub_;
	ros::Subscriber imu_sub_;
	tf::StampedTransform radar_to_imu_;
  ceres::CauchyLoss *doppler_loss_;
  ceres::ScaledLoss *imu_loss_;
	ceres::LocalParameterization* quat_param_;
	std::deque<Eigen::Quaterniond*> orientations_;
  std::deque<Eigen::Vector3d*> speeds_;
  std::deque<Eigen::Vector3d*> gyro_biases_;
  std::deque<Eigen::Vector3d*> accel_biases_;
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

  /** \brief uses ceres solver to estimate the body velocity from the input
    * point cloud
    * \param[out] the resultant body velocity
    * \param[out] the resultant velocity and covariance in ros message form
    */
  void GetVelocity(pcl::PointCloud<RadarPoint>::Ptr cloud,
                   double timestamp,
                   geometry_msgs::TwistWithCovarianceStamped &vel_out)
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
      //problem_->SetParameterBlockConstant(gyro_biases_.front()->data());
      //problem_->SetParameterBlockConstant(accel_biases_.front()->data());
		}
    // add latest parameter blocks and remove old ones if necessary
    std::vector<ceres::ResidualBlockId> residuals;
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);

		if (!ApplyMarginalization())
      LOG(ERROR) << "marginalization step failed";

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
                                      weight);

			// add residual block to ceres problem
      ceres::ResidualBlockId res_id =
				problem_->AddResidualBlock(doppler_cost_function,
                                   doppler_loss_,
                                   orientations_.front()->coeffs().data(),
                                	 speeds_.front()->data());
      residual_blks_.front().push_back(res_id);
    }

    // add imu cost only if there are more than 1 radar measurements in the queue
    if (timestamps_.size() >= 2)
    {
      std::vector<ImuMeasurement> imu_measurements =
					imu_buffer_.GetRange(timestamps_[1], timestamps_[0], true);

			ceres::CostFunction* imu_cost_func =
        	new GlobalImuVelocityCostFunction(timestamps_[1],
																			      timestamps_[0],
																			      imu_measurements,
																			      params_);

			ceres::ResidualBlockId res_id
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
			residual_blks_[1].push_back(res_id);
    }

		// solve the ceres problem and get result
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);

    LOG(INFO) << summary.FullReport();
    LOG(ERROR) << "   velocity: " << speeds_.front()->transpose();
    LOG(ERROR) << "orientation: " << orientations_.front()->coeffs().transpose();
    LOG(ERROR) << "  gyro bias: " << gyro_biases_.front()->transpose();
    LOG(ERROR) << " accel bias: " << accel_biases_.front()->transpose() << "\n\n";

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

    VLOG(2) << "covariance: \n" << covariance_matrix;
    */
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
    populateMessage(vel_out,covariance_matrix);
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
    Eigen::Vector3d g_vec = -sum_a / measurements.size();

    // set gravity vector magnitude
    params_.g_ = g_vec.norm();

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
    Eigen::Vector3d down_vec(0,0,1);
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

    Eigen::Quaterniond orientation_initial = Eigen::Quaterniond::Identity();
    // initial oplus increment
    Eigen::Vector4d dq;
    double halfnorm = 0.5 * increment.norm();
    QuaternionParameterization qp;
    dq.head(3) = qp.sinc(halfnorm) * 0.5 * increment;
    dq[3] = std::cos(halfnorm);

    orientation_initial = (Eigen::Quaterniond(dq) * orientation_initial);
    orientation_initial.normalize();

    orientations_.push_front(
        new Eigen::Quaterniond(orientation_initial));

    problem_->AddParameterBlock(orientations_.front()->coeffs().data(),4);
    problem_->SetParameterization(orientations_.front()->coeffs().data(), quat_param_);
    problem_->AddParameterBlock(speeds_.front()->data(),3);
    problem_->AddParameterBlock(gyro_biases_.front()->data(),3);
    problem_->AddParameterBlock(accel_biases_.front()->data(),3);
    problem_->SetParameterBlockConstant(gyro_biases_.front()->data());
    problem_->SetParameterBlockConstant(accel_biases_.front()->data());

    initialized_ = true;
    LOG(ERROR) << "initialized!";
    LOG(ERROR) << "initial orientation: " << orientation_initial.coeffs().transpose();
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
      if (marginalization_error_ptr_->num_residuals() == 0)
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

  /** \brief populate ros message with velocity and covariance
    * \param[out] vel the resultant ros message
    * \param[in] velocity the estimated velocity
    * \param[in] covariance the estimate covariance
    */
  void populateMessage(geometry_msgs::TwistWithCovarianceStamped &vel,
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

    Eigen::Vector3d velocity = *(speeds_.front());
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
	ImuParams params;
  RadarInertialVelocityReader* rv_reader = new RadarInertialVelocityReader(nh);

  //ros::spin();
	ros::AsyncSpinner spinner(4);
	spinner.start();
	ros::waitForShutdown();

  return 0;
}
