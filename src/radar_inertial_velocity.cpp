#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include  <glog/logging.h>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <ceres/ceres.h>
#include "CeresCostFunctions.h"
#include "DataTypes.h"
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

  RadarInertialVelocityReader(ros::NodeHandle nh, ImuParams params) 
  {
    nh_ = nh;
		params_ = params;
    imu_buffer_.SetTimeout(params_.frequency_);
		std::string radar_topic;
    std::string imu_topic;
    nh_.getParam("radar_topic", radar_topic);
    nh_.getParam("imu_topic", imu_topic);
    VLOG(2) << "radar topic: " << radar_topic;
    VLOG(2) << "imu topic: " << imu_topic;

    pub_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("/mmWaveDataHdl/vel",1);
    radar_sub_ = nh_.subscribe(radar_topic, 0, &RadarInertialVelocityReader::radarCallback, this);
		imu_sub_ = nh_.subscribe(imu_topic, 0, &RadarInertialVelocityReader::imuCallback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;
		initialized_ = false;

    window_size_ = 4;

    // set up ceres problem
    doppler_loss_ = new ceres::CauchyLoss(0.15);
    imu_loss_ = new ceres::CauchyLoss(0.1);

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

		imu_buffer_.AddMeasurement(new_meas);
	}

  void radarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
	  double timestamp = msg->header.stamp.toSec();
	  pcl::PointCloud<RadarPoint>::Ptr cloud(new pcl::PointCloud<RadarPoint>);
	  pcl::fromROSMsg(*msg, *cloud);
    
    // Reject clutter
    Declutter(cloud);
    
    Eigen::Vector3d coeffs = Eigen::Vector3d::Zero();
    
    if (cloud->size() < 10)
      LOG(WARNING) << "input cloud has less than 10 points, output will be unreliable";
    
    geometry_msgs::TwistWithCovarianceStamped vel_out;
    vel_out.header.stamp = msg->header.stamp;

    // Get velocity measurements
    auto start = std::chrono::high_resolution_clock::now();
    GetVelocity(cloud, coeffs, timestamp, vel_out);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    sum_time_ += elapsed.count();
    num_iter_++;
    std::cout << "execution time: " << sum_time_ / double(num_iter_) << std::endl;
    pub_.publish(vel_out);
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber radar_sub_;
	ros::Subscriber imu_sub_;
  ceres::CauchyLoss *doppler_loss_;
  ceres::CauchyLoss *imu_loss_;
  std::deque<Eigen::Vector3d*> velocities_;
  std::deque<double> timestamps_;
  std::deque<std::vector<ceres::ResidualBlockId>> residual_blks_;
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

	/** \brief uses initial IMU measurements to determine the magnitude of the
		* gravity vector and the initial attitude
		*/
	void InitializeImu()
	{
		double t0 = imu_buffer_.GetStartTime();
		std::vector<ImuMeasurement> measurements = 
					imu_buffer_.GetRange(t0, t0 + 0.1, false);

		// find gravity vector
		Eigen::Vector3d sum_a = Eigen::Vector3d::Zero();
		for (size_t i = 0; i < measurements.size(); i++)
			sum_a += measurements[i].a_;
		Eigen::Vector3d g_vec = -sum_a / measurements.size();

		// set gravity vector magnitude
		params_.g_ = g_vec.norm();

		// set initial orientation (attitude only)
		
		// set initialized to true
		initialized_ = true;
	}

  /** \brief uses ceres solver to estimate the body velocity from the input 
    * point cloud
    * \param[out] the resultant body velocity
    * \param[out] the resultant velocity and covariance in ros message form
    */
  void GetVelocity(pcl::PointCloud<RadarPoint>::Ptr cloud, 
                   Eigen::Vector3d &velocity, 
                   double timestamp,
                   geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {
		// if imu is not initialized, run initialization
		if (!initialized)
			InitializeImu();

    // add latest parameter block and remove old one if necessary
    velocities_.push_front(new Eigen::Vector3d());
    std::vector<ceres::ResidualBlockId> residuals;
    residual_blks_.push_front(residuals);
    timestamps_.push_front(timestamp);
    problem_->AddParameterBlock(velocities_.front()->data(),3);
    if (velocities_.size() >= window_size_)
    {
      for (int i = 0; i < residual_blks_.back().size(); i++)
        problem_->RemoveResidualBlock(residual_blks_.back()[i]);

      problem_->RemoveParameterBlock(velocities_.back()->data());
      delete velocities_.back();
      velocities_.pop_back();
      residual_blks_.pop_back();
      timestamps_.pop_back();
    }
    
    double min_intensity, max_intensity;
    getIntensityBounds(min_intensity,max_intensity,cloud);
    // add residuals on doppler readings
    for (int i = 0; i < cloud->size(); i++)
    {
      // calculate weight as normalized intensity
      double weight = (cloud->at(i).intensity - min_intensity)
                       / (max_intensity - min_intensity);
      Eigen::Vector3d target(cloud->at(i).x,
                             cloud->at(i).y,
                             cloud->at(i).z);
      ceres::CostFunction* doppler_cost_function = 
        new BodyVelocityCostFunction(cloud->at(i).doppler,
                                      target,
                                      weight);
      // add residual block to ceres problem
      ceres::ResidualBlockId res_id = problem_->AddResidualBlock(doppler_cost_function, 
                                                                 doppler_loss_, 
                                                                 velocity.data());
      residual_blks_.front().push_back(res_id);
    }
    
    // add residual on change in velocity if applicable
    if (timestamps_.size() >= 2)
    {
      double delta_t = timestamps_[0] - timestamps_[1];
      ceres::CostFunction* vel_change_cost_func = 
        new VelocityChangeCostFunction(delta_t);
      ceres::ResidualBlockId res_id = problem_->AddResidualBlock(vel_change_cost_func, 
                                                                 imu_loss_, 
                                                                 velocities_[0]->data(), 
                                                                 velocities_[1]->data());
      residual_blks_[1].push_back(res_id);
    }

// solve the ceres problem and get result
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);

    VLOG(3) << summary.FullReport();
    VLOG(2) << "velocity from ceres: " << velocities_.front()->transpose();

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
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Identity();
    populateMessage(vel_out,velocity,covariance_matrix);
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
    
    vel.twist.twist.linear.x = velocity[0];
    vel.twist.twist.linear.y = velocity[1];
    vel.twist.twist.linear.z = velocity[2];

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
        vel.twist.covariance[(i*6)+j] = covariance(i,j);
    }
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
	// need to add a way to read in IMU parameters from config file
	// perhaps using yaml-cpp
	ImuParams params;
  RadarInertialVelocityReader* rv_reader = new RadarInertialVelocityReader(nh, params);
  
  ros::spin();

  return 0;
}
