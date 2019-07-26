"""
Author:         Carl Stahoviak
Date Created:   July 25, 2019
Last Edited:    July 25, 2019

Description:
Base Estimator class for MLESAC Regression.

"""

import numpy as np
from goggles.radar_utilities import RadarUtilities

class dopplerMLESAC():

    def __init__(self, model):
        ## ascribe doppler velocity model (2D, 3D) to the class
        self.model = model
        self.utils = RadarUtilities()

        ## define MLESAC parameters
        self.sampleSize     = 3      # the minimum number of data values required to fit the model
        self.maxIterations  = 25     # the maximum number of iterations allowed in the algorithm
        self.maxDistance    = 0.15   # a threshold value for determining when a data point fits a model
        self.converge_thres = 10     # change in data log likelihood fcn required to indicate convergence

        self.report_scores = False
        self.ols_flag      = False

        self.param_vec_ = None      # body-frame velocity vector - to be estimated by MLESAC
        self.inliers    = None      # inlier data points
        self.scores     = None      # data log likelihood associated with each iteration
        self.iter       = None      # number of iterations until convergence

    ## model fit fcn
    def fit(self, data):
        self.param_vec_ = self.model.doppler2BodyFrameVelocity(data)
        print("fit: self.param_vec_ = " + str(self.param_vec_))
        return self

    ## distance(s) from data point(S) to model
    def distance(self, data):
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
        doppler_predicted = self.model.simulateRadarDoppler(self.param_vec_, \
            np.column_stack((radar_azimuth,radar_elevation)), eps, delta)

        eps_sq = np.square(np.subtract(doppler_predicted,radar_doppler))
        distances = np.sqrt(eps_sq)

        ## distance per data point (column vector)
        return distances
        # return np.squeeze(distances)

    ## evaluate the data log likelihood of the data given the model - P(evidence | model)
    def score(self, data):
        Ntargets = data.shape[0]
        p = data.shape[1]

        radar_doppler = data[:,0]
        radar_azimuth = data[:,1]
        radar_elevaton = data[:,2]

        doppler_predicted = self.model.simulateRadarDoppler(self.param_vec_, \
            np.column_stack((radar_azimuth,radar_elevaton)), \
            np.zeros((Ntargets,), dtype=np.float32), \
            np.zeros(((p-1)*Ntargets,), dtype=np.float32))

        # evaluate the data log-likelihood given the model
        eps_sq = np.square(np.subtract(doppler_predicted,radar_doppler))
        score = -1/(2*self.model.sigma_vr**2)*np.sum(eps_sq)

        return score

    def is_data_valid(self, data):
        if data.shape[0] != data.shape[1]:
            raise ValueError("data must be an 2x2 or 3x3 square matrix")

        p = data.shape[1]           # dimension of velocity vector

        if p == 2:
            radar_doppler = data[:,0]
            radar_azimuth = data[:,1]

            numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)

            if numAzimuthBins > 1:
                is_valid = True
            else:
                is_valid = False

        elif p == 3:
            radar_doppler = data[:,0]
            radar_azimuth = data[:,1]
            radar_elevation = data[:,2]

            numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)
            numElevBins = self.utils.getNumAzimuthBins(radar_elevation)

            if numAzimuthBins + numElevBins > 4:
                is_valid = True
            else:
                is_valid = False
        else:
            raise ValueError("data must be an Nx2 or Nx3 matrix")

        return is_valid
