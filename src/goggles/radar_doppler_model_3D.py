"""
Author:         Carl Stahoviak
Date Created:   Apr 22, 2019
Last Edited:    Apr 22, 2019

Description:

"""

import rospy
import numpy as np
from goggles.radar_utilities import RadarUtilities

class RadarDopplerModel3D:

    def __init__(self):
        self.utils = RadarUtilities()

        ## define radar parameters
        self.sigma_vr = 0.044               # [m/s]
        self.sigma_theta = 0.0426           # [rad]
        self.sigma_phi = self.sigma_theta   # [rad]

        ## define MLESAC parameters
        self.sampleSize     = 3      # the minimum number of data values required to fit the model
        self.maxIterations  = 25     # the maximum number of iterations allowed in the algorithm
        self.maxDistance    = 0.15   # a threshold value for determining when a data point fits a model
        self.converge_thres = 10     # change in data log likelihood fcn required to indicate convergence

    # defined for RANSAC - not used
    def fit(self, data):
        model = self.doppler2BodyFrameVelocity(data)
        return model

    # defined for RANSAC - not used
    def distance(self, data, model):
        Ntargets = data.shape[0]
        p = data.shape[1]

        # init distances vector
        distances = np.zeros((Ntargets,), dtype=np.float32)

        radar_doppler   = data[:,0]      # [m/s]
        radar_azimuth   = data[:,1]      # [rad]
        radar_elevation = data[:,2]      # [rad]

        ## do NOT corrupt measurements with noise
        eps = np.zeros((Ntargets,), dtype=np.float32)
        delta = np.zeros(((p-1)*Ntargets,), dtype=np.float32)

        ## radar doppler generative model
        doppler_predicted = self.simulateRadarDoppler(model, \
            np.column_stack((radar_azimuth,radar_elevation)), eps, delta)

        eps_sq = np.square(np.subtract(doppler_predicted,radar_doppler))
        distances = np.sqrt(eps_sq)

        ## distance per data point (column vector)
        return distances
        # return np.squeeze(distances)

    ## evaluate the data log likelihood of the data given the model - P(evidence | model)
    def score(self, data, model):
        Ntargets = data.shape[0]
        p = data.shape[1]

        radar_doppler = data[:,0]
        radar_azimuth = data[:,1]
        radar_elevaton = data[:,2]

        doppler_predicted = self.simulateRadarDoppler(model, \
            np.column_stack((radar_azimuth,radar_elevaton)), \
            np.zeros((Ntargets,), dtype=np.float32), \
            np.zeros(((p-1)*Ntargets,), dtype=np.float32))

        # evaluate the data log-likelihood given the model
        eps_sq = np.square(np.subtract(doppler_predicted,radar_doppler))
        score = -1/(2*self.sigma_vr**2)*np.sum(eps_sq)

        return score

    # inverse measurement model: measurements->model
    def doppler2BodyFrameVelocity(self, data):
        # data - a 3x3 matrix
        p = data.shape[1]

        radar_doppler = data[:,0]       # doppler velocity [m/s]
        theta         = data[:,1]       # azimuth angle column vector [rad]
        phi           = data[:,2]       # elevation angle column vector [rad]

        numAzimuthBins = self.utils.getNumAzimuthBins(theta)
        numElevBins = self.utils.getNumAzimuthBins(phi)

        # rospy.loginfo("doppler2BodyFrameVelocity: numAzimuthBins = %d", numAzimuthBins)
        # rospy.loginfo(['{0:5.4f}'.format(i) for i in radar_azimuth])    # 'list comprehension'

        if numAzimuthBins + numElevBins > 4:
           ## solve uniquely-determined problem for pair of targets (i,j)
            M = np.array([ [np.cos(theta[0])*np.cos(phi[0]), np.sin(theta[0])*np.cos(phi[0]), np.sin(phi[0])], \
                           [np.cos(theta[1])*np.cos(phi[1]), np.sin(theta[1])*np.cos(phi[1]), np.sin(phi[1])], \
                           [np.cos(theta[2])*np.cos(phi[2]), np.sin(theta[2])*np.cos(phi[2]), np.sin(phi[2])] ])

            b = np.array([ [radar_doppler[0]], \
                           [radar_doppler[1]], \
                           [radar_doppler[2]] ])

            model = np.squeeze(np.linalg.solve(M,b))
        else:
            model = float('nan')*np.ones((p,))

        return model

    # measurement generative (forward) model: model->measurements
    def simulateRadarDoppler(self, model, data, eps, delta):
        Ntargets = data.shape[0]
        radar_doppler = np.zeros((Ntargets,), dtype=np.float32)

        # unpack radar data
        radar_azimuth = data[:,0]
        radar_elevation = data[:,1]

        delta_theta = delta[:Ntargets]
        delta_phi = delta[-Ntargets:]

        for i in range(Ntargets):
            ## add measurement noise distributed as N(0,sigma_theta)
            theta = radar_azimuth[i] + delta_theta[i]

            ## add measurement noise distributed as N(0,sigma_phi)
            phi = radar_elevation[i] + delta_phi[i]

            ## add meaurement noise epsilon distributed as N(0,sigma_vr)
            radar_doppler[i] = model[0]*np.cos(theta)*np.cos(phi) + \
                model[1]*np.sin(theta)*np.cos(phi) + model[2]*np.sin(phi)

            ## add meaurement noise distributed as N(0,sigma_vr)
            radar_doppler[i] = radar_doppler[i] + eps[i]

        return radar_doppler


    def getBruteForceEstimate(self, radar_doppler, radar_azimuth):
        pass

    def getSimulatedRadarMeasurements(self, Ntargets, model, radar_angle_bins, \
                                        sigma_vr, debug=False):
        p = model.shape[0]

        # generate ptcloud of simulated targets
        ptcloud = self.generatePointcloud3D(Ntargets)

        radar_x = ptcloud[:,0]
        radar_y = ptcloud[:,1]
        radar_z = ptcloud[:,2]

        ## generate truth data
        radar_range = np.sqrt(radar_x**2 + radar_y**2 + radar_z**2)
        true_azimuth = np.arctan(np.divide(radar_y,radar_x))
        true_elevation = np.arcsin(np.divide(radar_z,radar_range))

        ## init simulated data vectors
        radar_azimuth = np.zeros((Ntargets,), dtype=np.float32)
        radar_elevation = np.zeros((Ntargets,), dtype=np.float32)

        # bin azimuth and elevation data
        for i in range(Ntargets):
            bin_idx = (np.abs(radar_angle_bins - true_azimuth[i])).argmin()
            ## could additionally add Gaussian noise here
            radar_azimuth[i] = radar_angle_bins[bin_idx]

            bin_idx = (np.abs(radar_angle_bins - true_elevation[i])).argmin()
            ## could additionally add Gaussian noise here
            radar_elevation[i] = radar_angle_bins[bin_idx]

        ## define AGWN vector for doppler velocity measurements
        if debug:
            eps = np.ones((Ntargets,), dtype=np.float32)*sigma_vr
        else:
            eps = np.random.normal(0,sigma_vr,(Ntargets,))

        ## get true radar doppler measurements
        true_doppler = self.simulateRadarDoppler(model, np.column_stack((true_azimuth,true_elevation)), \
            np.zeros((Ntargets,), dtype=np.float32), np.zeros(((p-1)*Ntargets,), dtype=np.float32))

        # get noisy radar doppler measurements
        radar_doppler =  self.simulateRadarDoppler(model, np.column_stack((radar_azimuth,radar_elevation)), \
            eps, np.zeros(((p-1)*Ntargets,), dtype=np.float32))

        data_truth = np.column_stack((true_doppler,true_azimuth,true_elevation))
        data_sim = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))

        return data_truth, data_sim

    def generatePointcloud3D(self, Ntargets):
        ## need to update this function to only generate targets within a
        ## specifed azimuth and elevation FOV - these ranges taken as inputs
        ## to this function

        min_x = 0
        max_x = 10
        ptcloud_x = (max_x - min_x)*np.random.random((Ntargets,)) + min_x

        min_y = -10
        max_y = 10
        ptcloud_y = (max_y - min_y)*np.random.random((Ntargets,)) + min_y

        min_z = -10
        max_z = 10
        ptcloud_z = (max_z - min_z)*np.random.random((Ntargets,)) + min_z

        pointcloud = np.column_stack((ptcloud_x,ptcloud_y,ptcloud_z))

        return pointcloud
