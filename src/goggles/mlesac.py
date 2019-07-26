#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   July 23, 2019
Last Edited:    July 24, 2019

Description:

"""

import time
import numpy as np
# import scipy as sp
from goggles.radar_utilities import RadarUtilities
from goggles.radar_doppler_model_2D import RadarDopplerModel2D
from goggles.radar_doppler_model_3D import RadarDopplerModel3D
from goggles.base_estimator_mlesac import dopplerMLESAC

class MLESAC:

    def __init__(self, base_estimator):
        self.estimator_ = base_estimator

    def mlesac(self, data):

        Ntargets = data.shape[0]    # number of data points
        p = data.shape[1]           # dimension of velocity vector

        if self.estimator_.sampleSize != p:
            raise ValueError("radar model does NOT match column dimension of data")

        # if p == 2:
        #     radar_doppler = data[:,0]
        #     radar_azimuth = data[:,1]
        # elif p == 3:
        #     radar_doppler = data[:,0]
        #     radar_azimuth = data[:,1]
        #     radar_elevation = data[:,2]
        # else:
        #     raise ValueError("data must be an Nx2 or Nx3 matrix")

        bestScore = -np.inf
        bestInliers = []
        bestModel = []
        scores = []

        dll_incr = np.inf           # increase in data log likelihood function
        iter = 0                    # algorithm iteration Number

        while np.abs(dll_incr) > self.estimator_.converge_thres and \
            iter < self.estimator_.maxIterations:

            ## randomly sample from data
            idx = np.random.randint(Ntargets,high=None,size=(p,))
            sample = data[idx,:]

            is_valid = self.estimator_.is_data_valid(sample)
            if is_valid:
                ## estimate model parameters from sampled data points
                param_vec_temp = self.estimator_.param_vec_
                self.estimator_.fit(sample)

                ## score the model - evaluate the data log likelihood fcn
                score = self.estimator_.score(data)

                if score > bestScore:
                    ## this model better explains the data
                    distances = self.estimator_.distance(data)

                    dll_incr = score - bestScore    # increase in data log likelihood fcn
                    bestScore = score
                    bestInliers = np.nonzero((distances < self.estimator_.maxDistance))

                    if self.estimator_.report_scores:
                        scores.append(score)

                    # evaluate stopping criteria - not yet used
                    Ninliers = sum(bestInliers)
                    w = Ninliers/Ntargets
                    k = np.log(1-0.95)*np.log(1-w^2)
                else:
                    ## candidate param_vec_ did NOT have a higher score
                    self.estimator_.param_vec_ = param_vec_temp

                iter+=1
                # print("iter = " + str(iter) + "\tscore = " + str(score))
            else:
                # do nothing - cannot derive a valid model fromtargets in
                # the same azimuth/elevation bins

                # print("mlesac: INVALID DATA SAMPLE")
                pass

        ## get OLS solution on inlier set
        if self.estimator_.ols_flag:
            pass
            # model_ols = sp.optimize.least_squares()
        else:
            model_ols = float('nan')*np.ones((p,))

        self.estimator_.inliers = bestInliers
        self.estimator_.scores = np.array(scores)
        self.estimator_.iter = iter
        return self


def test(model):
    # init instance of base estimator dopplerMLESAC class
    base_estimator = dopplerMLESAC(model)
    mlesac = MLESAC(base_estimator)

    ## outlier std deviation
    sigma_vr_outlier = 1.5

    radar_angle_bins = np.genfromtxt('../../data/1642_azimuth_bins.csv', delimiter=',')

    ## simulated 'true' platform velocity range
    min_vel = -2.5      # [m/s]
    max_vel = 2.5       # [m/s]

    ## number of simulated targets
    Ninliers = 70
    Noutliers = 35

    ## generate truth velocity vector
    velocity = (max_vel-min_vel)*np.random.random((base_estimator.sampleSize,)) + min_vel

    ## create noisy INLIER  simulated radar measurements
    _, inlier_data = model.getSimulatedRadarMeasurements(Ninliers, \
        velocity,radar_angle_bins,model.sigma_vr)

    ## create noisy OUTLIER simulated radar measurements
    _, outlier_data = model.getSimulatedRadarMeasurements(Noutliers, \
        velocity,radar_angle_bins,sigma_vr_outlier)

    ## combine inlier and outlier data sets
    Ntargets = Ninliers + Noutliers
    radar_doppler = np.concatenate((inlier_data[:,0],outlier_data[:,0]),axis=0)
    radar_azimuth = np.concatenate((inlier_data[:,1],outlier_data[:,1]),axis=0)
    radar_elevation = np.concatenate((inlier_data[:,2],outlier_data[:,2]),axis=0)

    radar_data = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))
    start_time = time.time()
    # model_mlesac, inliers, _, _ = mlesac.mlesac(radar_data)
    mlesac.mlesac(radar_data)
    end_time = time.time()
    model_mlesac = mlesac.estimator_.param_vec_
    inliers = mlesac.estimator_.param_vec_

    print("\nMLESAC Velocity Profile Estimation:\n")
    print("True Velocity Vector\t MLESAC Estimated Velocity Vector")
    print(str.format('{0:.4f}',velocity[0]) + "\t " + str.format('{0:.4f}',model_mlesac[0]))
    print(str.format('{0:.4f}',velocity[1]) + "\t " + str.format('{0:.4f}',model_mlesac[1]))
    print(str.format('{0:.4f}',velocity[2]) + "\t " + str.format('{0:.4f}',model_mlesac[2]))

    rmse_mlesac = np.sqrt(np.mean(np.square(velocity - model_mlesac)))
    # rmse_ols = np.sqrt(np.mean(np.square(velocity - model_ols)))

    print("\nRMSE (MLESAC) = " + str.format('{0:.4f}',rmse_mlesac) + " m/s")
    # print("\nRMSE (OLS) = " + str.format('{0:.4f}',rmse_ols) + " m/s")
    print("Execution Time = %s" % (end_time-start_time))

def test_montecarlo(model):
    pass

if __name__=='__main__':
    # model = RadarDopplerModel2D()
    model = RadarDopplerModel3D()
    test(model)
