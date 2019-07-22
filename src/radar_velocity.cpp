#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
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
#include <CeresCostFunction.h>
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

    pub_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("/mmWaveDataHdl/vel",1);
    sub_ = nh_.subscribe(radar_topic, 0, &RadarVelocityReader::callback, this);
    min_range_ = 0.5;
    sum_time_ = 0.0;
    num_iter_ = 0;
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr& msg)
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
    GetVelocityCeres(cloud, coeffs, vel_out);
    //GetVelocityIRLS(cloud, coeffs, 100, 1.0e-10, vel_out);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    sum_time_ += elapsed.count();
    VLOG(2) << "execution time: " << sum_time_ / double(num_iter_);
    pub_.publish(vel_out);
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
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
    * \param[out] the resultant body velocity
    * \param[out] the resultant velocity and covariance in ros message form
    */
  void GetVelocityCeres(pcl::PointCloud<RadarPoint>::Ptr cloud, 
                   Eigen::Vector3d &velocity, 
                   geometry_msgs::TwistWithCovarianceStamped &vel_out)
  {

    // set up ceres problem
    ceres::Problem::Options prob_options;
    std::shared_ptr<ceres::Problem> problem;
    ceres::Solver::Options solver_options;
    
    prob_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    solver_options.num_threads = 4;
    solver_options.max_num_iterations = 300;
    solver_options.update_state_every_iteration = true;
    solver_options.function_tolerance = 1e-10;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    problem.reset(new ceres::Problem(prob_options));

    // add parameter block
    problem->AddParameterBlock(velocity.data(), 3);  
    
    double min_intensity, max_intensity;
    getIntensityBounds(min_intensity,max_intensity,cloud);

    ceres::CauchyLoss *loss_func = new ceres::CauchyLoss(0.15);
    for (int i = 0; i < cloud->size(); i++)
    {
      // calculate weight as normalized intensity
      double weight = (cloud->at(i).intensity - min_intensity)
                       / (max_intensity - min_intensity);
      Eigen::Vector3d target(cloud->at(i).x,
                             cloud->at(i).y,
                             cloud->at(i).z);
      // set up cost function
      ceres::CostFunction* cost_function = 
        new ceres::AutoDiffCostFunction<BodyVelocityCostFunction<double>, 1, 3>(
          new BodyVelocityCostFunction<double>(cloud->at(i).doppler,
                                       target,
                                       weight));
      // add residual block to ceres problem
      problem->AddResidualBlock(cost_function, loss_func, velocity.data());
    }

    // solve the ceres problem and get result
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, problem.get(), &summary);

    VLOG(3) << summary.FullReport();
    VLOG(2) << "velocity from ceres: " << velocity.transpose();

    // get estimate covariance
    ceres::Covariance::Options cov_options;
    cov_options.num_threads = 1;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(cov_options);

    std::vector<std::pair<const double*, const double*>> cov_blks;
    cov_blks.push_back(std::make_pair(velocity.data(),velocity.data()));

    covariance.Compute(cov_blks, problem.get());
    Eigen::Matrix3d covariance_matrix;
    covariance.GetCovarianceBlock(velocity.data(), 
                                  velocity.data(), 
                                  covariance_matrix.data());

    VLOG(2) << "covariance: \n" << covariance_matrix;
    
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
  RadarVelocityReader* rv_reader = new RadarVelocityReader(nh);
  
  ros::spin();

  return 0;
}