#pragma once
#define PCL_NO_PRECOMPILE
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <iomanip>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <deque>

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

struct ImuMeasurement
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Eigen::Vector3d g_;    // gyro reading
	Eigen::Vector3d a_;    // accelerometer reading
	Eigen::Quaterniond q_; // orientation from AHRS
	double t_;             // timestamp
};

struct ImuParams
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	double frequency_; // imu frequency
	double g_;         // gravity magnitude
	double g_max_;	   // max gyro reading
	double a_max_;     // max acceleration
	double sigma_g_;   // gyro noise prior
	double sigma_a_;   // accelerometer noise prior
	double sigma_b_g_; // gyro bias noise prior
	double sigma_b_a_; // accelerometer bias noise prior
	double b_a_tau_;   // accelerometir bias random walk parameter
	double invert_yaw_; // is the yaw estimated by the ahrs inverted?
	Eigen::Matrix3d ahrs_to_imu_; // ahrs to imu frame transform
};

class ImuBuffer
{
	public:
		size_t size()
		{
			return measurements_.size();
		}

		void SetTimeout(double imu_freq)
		{
			timeout_ = std::chrono::milliseconds(50 * int((1.0 / imu_freq) * 1000.0));
		}

		double GetStartTime()
		{
			return measurements_.front().t_;
		}

		double GetEndTime()
		{
			return measurements_.back().t_;
		}
		
		// add a new measurement to the buffer
		bool AddMeasurement(ImuMeasurement &meas)
		{
			if (measurements_.size() > 1 && meas.t_ < GetEndTime())
			{
				LOG(ERROR) << "received imu measurement's timestamp (" 
									 << meas.t_ << ") is less than that of the"
									 << " previously received measurement ("
									 << measurements_.back().t_ << ")";
				return false;
			}
			{
				std::lock_guard<std::mutex> lk(mtx_);
				measurements_.push_back(meas);
			}
			cv_.notify_one();
			return true;
		}

		bool WaitForMeasurements()
		{
			std::unique_lock<std::mutex> lk(mtx_);
			if (!cv_.wait_for(lk,
												timeout_,
												[this]{return measurements_.size() > 0;}))
			{
				LOG(ERROR) << "waiting for imu measurements has failed";
				return false;
			}
			return true;
		}

		// get a range of measurements from the buffer
		// returns a vector of ImuMeasurements
		// the first element in the vector will have a timestamp less than
		// or equal to t0 and the last element will have a timestamp 
		// greater than or equal to t1
		// if t1 is greater than the most recent timestamp in the buffer
		// execution will block either until up-to-date measurements are
		// available or 10 times the imu rate has elapsed
		std::vector<ImuMeasurement> GetRange(double t0, double t1, bool delete_old)
		{
			std::vector<ImuMeasurement> meas_range;
			
			if (t0 < GetStartTime())
				LOG(ERROR) << std::fixed << std::setprecision(3)
									 << "start time of requested range ("
									 << t0 << ") is less than the timestamp"
									 << " of the first measurement (" 
									 << measurements_.front().t_ << ")";
			
			// block execution until up-to-date imu measurements are available
			std::unique_lock<std::mutex> lk(mtx_);
			if (!cv_.wait_for(lk, 
											  timeout_,
											  [&t1,this]{return t1 <= GetEndTime();}))
			{
				LOG(ERROR) << std::fixed << std::setprecision(3)
									 << "waiting for up to date imu measurements has failed\n"
									 << "             requested t1: " << t1 << '\n'
									 << "most recent imu timestamp: " << GetEndTime();
				return meas_range;
			}
			
			// find index of measurement immediately before (or equal to) t0
			size_t start_index = 0;
			while (measurements_[start_index + 1].t_ < t0)
				start_index++;

			if (measurements_[start_index].t_ > t0)
				LOG(ERROR) << "timestamp of start index is greater than t0";

			// find index of measurement immediately after (or equal to) t1
			size_t end_index = measurements_.size() - 1;
			while (measurements_[end_index - 1].t_ > t1)
				end_index--;

			if (measurements_[end_index].t_ < t1)
				LOG(ERROR) << "timestamp of end index is less than t1";
			
			// add measurement range to return vector
			for (size_t i = start_index; i <= end_index; i++)
				meas_range.push_back(measurements_[i]);

			// delete old measurements (assuming they won't be needed again)
			// keep the newest 5 measurements
			if (delete_old)
			{
				start_index = end_index - 5;
				while (start_index > 0)
				{
					measurements_.pop_front();
					start_index--;
				}
			}

			return meas_range;
		}

	private:
		std::deque<ImuMeasurement> measurements_;
		std::condition_variable cv_;
		std::mutex mtx_;
		std::chrono::milliseconds timeout_;
};
