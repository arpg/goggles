#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   Apr 21, 2019
Last Edited:    Apr 21, 2019

Task: To estimate the 2D or 3D body-frame velocity vector of the sensor
platfrom given input data from a single radar (/mmWaveDataHdl/RScan topic).
The velocity estimation scheme takes the following approach:

1. Near-field targets are removed from the target list. Many of these targets
are artifacts of antenna interference at the senor origin, and are not
representative of real targets in the scene. These near-field targets also exist
in the zero-doppler bin and thus would currupt the quality of the velocity
estimate.
2. A RANSAC (or MLESAC) outlier rejection method is used to filter targets that
can be attributed to noise or dynamic targets in the environment. RANSAC
generates an inlier set of targets and a first-pass velocity estimate derived
from the inlier set.
3. Orthogonal Distance Regression (ODR) is seeded with the RANSAC velocity
estimate and generates a final estimate of the body frame linear velocity
components.

Implementation:
- The goal is for the VelocityEstimator class (node) to be model dependent
(e.g. 2D or 3D). In both cases the node is subcribed to the same topic
(/mmWaveDataHdl/RScan), and publishes a TwistWithCovarianceStamped message.
- I will need to write to separate classes (RadarDopplerModel2D and
RadarDopplerModel3D) that both define the same methods (e.g.
doppler2BodyFrameVelocity, simulateRadarDoppler, etc.) such that the
VelocityEstimator class is composed of either one of these models. This can be
thought of as designing an "implied interface" to the RadarDopplerModel subset
of classes.
- Additionally, a RadarUtilities class should be implemetned to allow access to
other functions that are model-independent.

"""

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistWithCovarianceStamped
from geometry_msgs.msg import TwistStamped

import numpy as np
from sklearn.linear_model import RANSACRegressor
from goggles.mlesac import MLESAC
from goggles.radar_doppler_model_2D import RadarDopplerModel2D
from goggles.radar_doppler_model_3D import RadarDopplerModel3D
from goggles.base_estimator import dopplerRANSAC
from goggles.base_estimator_mlesac import dopplerMLESAC
from goggles.orthogonal_distance_regression import OrthogonalDistanceRegression2D
from goggles.radar_utilities import RadarUtilities
import csv

WRITE_DATA = False

class VelocityEstimator():

    def __init__(self, model):
        if WRITE_DATA:
            csv_file = open('bruteForce.csv', 'a')
            self.writer = csv.writer(csv_file, delimiter=',')

        ## prescribe velocity estimator model {2D, 3D} and utils class
        self.model = model
        self.odr   = odr=OrthogonalDistanceRegression2D()   # change to full 2D/3D compatible ODR in future
        self.utils = RadarUtilities()
        self.type  = rospy.get_param('~type')

        ## instantiate mlesac object with base_estimator_mlesac class object'
        self.base_estimator = dopplerMLESAC(model)
        self.mlesac = MLESAC(self.base_estimator)

        ns = rospy.get_namespace()
        rospy.loginfo("INIT: namespace = %s", ns)

        ## init subscriber
        self.mmwave_topic = rospy.get_param('~mmwave_topic')
        self.radar_sub = rospy.Subscriber(self.mmwave_topic, PointCloud2, self.ptcloud_cb)
        rospy.loginfo("INIT: VelocityEstimator Node subcribed to: " + self.mmwave_topic)

        ## init publisher
        twist_topic = 'goggles'
        self.twist_mlesac_pub = rospy.Publisher(ns + twist_topic, TwistStamped, queue_size=10)
        # self.twist_bf_pub = rospy.Publisher(ns + twist_topic +'_bf', TwistWithCovarianceStamped, queue_size=10)
        # self.twist_mlesac_pub = rospy.Publisher(ns + twist_topic +'_mlesac', TwistStamped, queue_size=10)
        # self.twist_odr_pub = rospy.Publisher(ns + twist_topic +'_odr', TwistWithCovarianceStamped, queue_size=10)

        ## define filtering threshold parameters - taken from velocity_estimation.m
        self.azimuth_thres   = rospy.get_param('~azimuth_thres')
        self.elevation_thres = rospy.get_param('~elevation_thres')
        self.range_thres     = rospy.get_param('~range_thres')
        self.intensity_thres = rospy.get_param('~intensity_thres')
        self.thresholds      = np.array([self.azimuth_thres, self.intensity_thres, \
                                         self.range_thres, self.elevation_thres])

        rospy.loginfo("INIT: " + self.mmwave_topic + " azimuth_thres = " + str(self.azimuth_thres))
        rospy.loginfo("INIT: " + self.mmwave_topic + " elevation_thres = " + str(self.elevation_thres))
        rospy.loginfo("INIT: " + self.mmwave_topic + " range_thres = " + str(self.range_thres))
        rospy.loginfo("INIT: " + self.mmwave_topic + " intensity_thres = " + str(self.intensity_thres))

        # use the ODR estimate? (if not, publish ransac estimate)
        self.odr_flag = rospy.get_param('~odr_flag')

        rospy.loginfo("INIT: VelocityEstimator Node Initialized")

    def ptcloud_cb(self, msg):
        rospy.loginfo("GOT HERE: ptcloud_cb")
        # rospy.loginfo("Messaged recieved on: " + rospy.get_namespace())
        pts_list = list(pc2.read_points(msg, field_names=["x", "y", "z", "intensity", "range", "doppler"]))
        pts = np.array(pts_list)

        pts[:,1] = -pts[:,1]    ## ROS standard coordinate system Y-axis is left, NED frame Y-axis is to the right
        pts[:,2] = -pts[:,2]    ## ROS standard coordinate system Z-axis is up, NED frame Z-axis is down

        ## pts.shape = (Ntargets, 6)
        if pts.shape[0] < self.base_estimator.sample_size:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from less than 2 targets
            rospy.logwarn("ptcloud_cb: EMPTY RADAR MESSAGE")
        else:
            # rospy.loginfo("\n")
            # rospy.loginfo("New Scan")
            # rospy.loginfo("Ntargets = %d", pts.shape[0])
            self.estimate_velocity(pts, msg)

    def estimate_velocity(self, pts, radar_msg):
        rospy.loginfo("GOT HERE: estimate_velocity")
        Ntargets = pts.shape[0]
        # rospy.loginfo("Ntargets = " + str(Ntargets))

        ## create target azimuth vector (in radians)
        azimuth = np.arctan(np.divide(pts[:,1],pts[:,0]))

        if self.model.min_pts == 2:
            elevation = float('nan')*np.ones((Ntargets,))
        elif self.model.min_pts == 2:
            radar_xy = np.sqrt(np.square(pts[:,0]) + np.square(pts[:,1]))
            elevation = np.arctan(np.divide(pts[:,2],radar_xy))
        else:
            rospy.logerr("velocity_estimator_node main(): ESTIMATOR TYPE IMPROPERLY SPECIFIED")

        data_AIRE = data_AIRE = np.column_stack((azimuth, pts[:,3], pts[:,4], elevation))
        idx_AIRE = self.utils.AIRE_filtering(data_AIRE, self.thresholds)
        # rospy.loginfo("Ntargets_valid = %d", idx_AIR.shape[0])

        ## define pre-filtered radar data for further processing
        radar_intensity = pts[idx_AIRE,3]
        radar_range     = pts[idx_AIRE,4]
        radar_doppler   = pts[idx_AIRE,5]
        radar_azimuth   = azimuth[idx_AIRE]
        radar_elevation = elevation[idx_AIRE]

        Ntargets_valid = radar_doppler.shape[0]
        # rospy.loginfo("Ntargets_valid = " + str(Ntargets_valid))

        if Ntargets_valid < self.base_estimator.sample_size:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from less than 2 targets
            rospy.logwarn("estimate_velocity: < %d TARGETS AFTER AIR THRESHOLDING" % self.base_estimator.sample_size)
        else:
            # rospy.loginfo("Nbins = %d", self.utils.getNumAzimuthBins(radar_azimuth))
            # rospy.loginfo(['{0:5.4f}'.format(i) for i in radar_azimuth])    # 'list comprehension'

            ## get brute-force estimate
            # model_bruteforce, _ = self.model.getBruteForceEstimate(radar_doppler, radar_azimuth)
            # rospy.loginfo("model_bruteforce = " + str(model_bruteforce))

            if WRITE_DATA:
                self.writer.writerow(model_bruteforce.tolist())

            ## get MLESAC estimate + inlier set
            radar_data = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))
            self.mlesac.mlesac(radar_data)
            model_mlesac = self.mlesac.estimator_.param_vec_
            # rospy.loginfo("model_mlesac = " + str(model_mlesac.T))

            ## get RANSAC estimate + inlier set
            # self.ransac.fit(np.array([radar_azimuth]).T, np.array([radar_doppler]).T)
            # model_ransac = np.squeeze(self.ransac.estimator_.param_vec_)
            # inlier_mask = self.ransac.inlier_mask_
            # # outlier_mask = np.logical_not(inlier_mask)

            intensity_inlier = radar_intensity[self.mlesac.inliers_]
            doppler_inlier   = radar_doppler[self.mlesac.inliers_]
            azimuth_inlier   = radar_azimuth[self.mlesac.inliers_]
            elevation_inlier = radar_elevation[self.mlesac.inliers_]

            ## get ODR estimate
            if self.odr_flag:
                Ntargets_inlier = doppler_inlier.shape[0]
                # rospy.loginfo("Ntargets_inlier = " + str(Ntargets_inlier))

                ## get ODR estimate
                weights = (1/self.odr.sigma_vr)*np.ones((Ntargets_inlier,), dtype=np.float32)
                delta = np.random.normal(0,self.odr.sigma_theta, \
                    ((self.base_estimator.sample_size-1)*Ntargets_inlier,))
                model_odr = self.odr.odr( doppler_inlier, azimuth_inlier, self.odr.d, \
                    model_mlesac, delta, weights )
                # rospy.loginfo("model_odr = " + str(model_odr))

            ## publish velocity estimate

            # if np.isnan(model_bruteforce[1]):
            #     rospy.logwarn("estimate_velocity: BRUTEFORCE VELOCITY ESTIMATE IS NANs")
            # else:
            #     velocity_estimate = -model_bruteforce
            #     self.publish_twist_estimate(velocity_estimate, radar_msg, type='bruteforce')

            if np.isnan(model_mlesac[0]):
                rospy.logwarn("estimate_velocity: RANSAC VELOCITY ESTIMATE IS NANs")
            else:
                if self.type == '2D':
                    velcity_estimate = np.stack((-model_mlesac,np.zeros((1,))))
                elif self.type =='3D':
                    velocity_estimate = -model_mlesac
                self.publish_twist_estimate(velocity_estimate, radar_msg, type='mlesac')

            # if np.isnan(model_odr[1]):
            #     rospy.logwarn("estimate_velocity: ODR VELOCITY ESTIMATE IS NANs")
            # else:
            #     velocity_estimate = -model_odr
            #     self.publish_twist_estimate(velocity_estimate, radar_msg, type='odr')

    def publish_twist_estimate(self, velocity_estimate, radar_msg, type=None):

        if type == 'bruteforce' or type == 'mlesac':
            twist_estimate = TwistStamped()

            twist_estimate.twist.linear.x = velocity_estimate[0]
            twist_estimate.twist.linear.y = velocity_estimate[1]
            twist_estimate.twist.linear.z = velocity_estimate[2]
        elif type == 'odr':
            twist_estimate = TwistWithCovarianceStamped()

            twist_estimate.twist.twist.linear.x = velocity_estimate[0]
            twist_estimate.twist.twist.linear.y = velocity_estimate[1]
            twist_estimate.twist.twist.linear.z = velocity_estimate[2]
        else:
            rospy.logerr("publish_twist_estimate: CANNOT PUBLISH TWIST MESSAGE ON UNSPECIFIED TOPIC")

        twist_estimate.header.stamp = radar_msg.header.stamp
        twist_estimate.header.frame_id = "base_link"

        if type == 'bruteforce':
            self.twist_bf_pub.publish(twist_estimate)
        elif type == 'mlesac':
            self.twist_mlesac_pub.publish(twist_estimate)
        elif type == 'odr':
            self.twist_odr_pub.publish(twist_estimate)
        else:
            rospy.logerr("publish_twist_estimate: CANNOT PUBLISH TWIST MESSAGE ON UNSPECIFIED TOPIC")

def main():
    ## anonymous=True ensures that your node has a unique name by adding random numbers to the end of NAME
    rospy.init_node('velocity_estimator_node')

    if WRITE_DATA:
        csv_file = open('bruteForce.csv', 'w+')
        csv_file.close()

    type = rospy.get_param('~type')
    if type == '2D':
        model = RadarDopplerModel2D()
    elif type =='3D':
        model = RadarDopplerModel3D()
    else:
        rospy.logerr("velocity_estimator_node main(): ESTIMATOR TYPE IMPROPERLY SPECIFIED")


    # use composition to ascribe a model to the VelocityEstimator class
    # velocity_estimator = VelocityEstimator(model=RadarDopplerModel2D(), \
    #                                        odr=OrthogonalDistanceRegression2D())
    velocity_estimator = VelocityEstimator(model=model)

    rospy.loginfo("End of main()")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        sys.exit()

if __name__ == '__main__':
    main()
