#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   Apr 28, 2019
Last Edited:    Apr 28, 2019

Description:

"""

from __future__ import division

import time
import rospy
import numpy as np
from numpy.linalg import inv, multi_dot
from scipy.linalg import block_diag
from scipy.linalg.blas import sgemm, sgemv
from goggles.radar_doppler_model_2D import RadarDopplerModel2D
from goggles.radar_doppler_model_3D import RadarDopplerModel3D
from goggles.base_estimator_mlesac import dopplerMLESAC
from goggles.mlesac import MLESAC

class OrthogonalDistanceRegression():

    def __init__(self, model, converge_thres=0.0005, max_iter=50, debug=False):
        self.model = model                      # radar Doppler model (2D or 3D)
        self.converge_thres = converge_thres    # ODR convergence threshold on step size s
        self.maxIterations = max_iter           # max number of ODR iterations
        self.debug = debug                      # for comparison to MATLAB implementation

        self.param_vec_   = None    # body-frame velocity vector - to be estimated by ODR
        self.covariance_  = None    # covariance of parameter estimate, shape (p,p)
        self.iter_        = 0       # number of iterations till convergence

        self.s_ = 10*np.ones((model.min_pts,), dtype=np.float32)    # step size scale factor

    def odr(self, data, beta0, weights, get_covar):
        """
        d - error variance ratio := sigma_vr / sigma_theta
        """
        ## unpack radar data (into colum vectors)
        radar_doppler   = data[:,0]
        radar_azimuth   = data[:,1]
        radar_elevation = data[:,2]

        ## dimensionality of data
        Ntargets = data.shape[0]
        p = beta0.shape[0]
        m = self.model.d.shape[0]

        # [ S, T ] = self.getScalingMatrices()
        S = np.diag(self.s_)                         # s scaling matrix - 10 empirically chosen
        T = np.eye(Ntargets*m, dtype=np.float32)    # t scaling matrix
        alpha = 1                                   # Lagrange multiplier

        if p == 2:
            ## init delta vector
            delta0 = np.random.normal(0,self.model.sigma,(Ntargets,)).astype(np.float32)

            ## construct weighted diagonal matrix D
            D = np.diag(np.multiply(weights,d)).astype(np.float32)

        elif p == 3:
            ## init delta vector - "interleaved" vector
            delta0_theta = np.random.normal(0,self.model.sigma[0],(Ntargets,)).astype(np.float32)
            delta0_phi   = np.random.normal(0,self.model.sigma[1],(Ntargets,)).astype(np.float32)
            delta0 = np.column_stack((delta0_theta, delta0_phi))
            delta0 = delta0.reshape((m*Ntargets,))

            ## construct weighted block-diagonal matrix D
            D_i = np.diag(self.model.d).astype(np.float32)
            Drep = D_i.reshape(1,m,m).repeat(Ntargets,axis=0)
            Dblk = block_diag(*Drep)
            weights_diag = np.diag(np.repeat(weights,m)).astype(np.float32)
            D = np.matmul(weights_diag,Dblk)

        else:
            rospy.logerr("odr: initial guess must be a 2D or 3D vector")

        ## construct E matrix - E = D^2 + alpha*T^2 (ODR-1987 Prop. 2.1)
        E = np.matmul(D,D) + alpha*np.matmul(T,T)
        Einv = inv(E)

        ## initialize
        beta = beta0
        delta = delta0
        s = np.ones((p,), dtype=np.float32)

        self.iter_ = 1
        while np.linalg.norm(s) > self.converge_thres:

            if p == 2:
                G, V, M = self.getJacobian2D(data[:,1], delta, beta, weights, E)

            elif p ==3:
                G, V, M = self.getJacobian3D(data[:,1:3], delta, beta, weights, E)

            else:
                rospy.logerr("odr: initial guess must be a 2D or 3D vector")

            doppler_predicted = self.model.simulateRadarDoppler(beta, \
                    np.column_stack((data[:,1],data[:,2])), \
                    np.zeros((Ntargets,), dtype=np.float32), delta)

            ## update epsilon
            eps = np.subtract(doppler_predicted, radar_doppler)

            ## defined to reduce the number of times certain matrix products are computed
            prod1 = np.matmul(D,delta)
            prod2 = multi_dot([V,Einv,prod1])

            ## form the elements of the linear least squares problem
            Gbar = np.matmul(M,G)
            y = np.matmul(-M,np.subtract(eps,prod2))

            ## Compute step s via QR factorization of Gbar
            Q,R = np.linalg.qr(Gbar,mode='reduced')
            s = np.squeeze(np.linalg.solve(R,np.matmul(Q.T,y)))

            # t = -Einv*(V'*M^2*(eps + G*s - V*Einv*D*delta) + D*delta)
            t = np.matmul(-Einv, np.add( multi_dot( [V.T, np.matmul(M,M), \
                    eps+np.matmul(G,s)-prod2] ), prod1 ))

            # use s and t to iteratively update beta and delta, respectively
            beta = beta + np.matmul(S,s)
            delta = delta + np.matmul(T,t)

            self.iter_ += 1
            if self.iter_ > self.maxIterations:
                rospy.loginfo('ODR: max iterations reached')
                break

        self.param_vec_ = beta
        if get_covar:
            self.getCovariance( Gbar, D, eps, delta, weights )
        else:
            self.covariance_ = float('nan')*np.ones((p,))

        return


    def getJacobian2D(self, X, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special properites
        V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix
        """

        Ntargets = X.shape[0]   # X is a column vector of azimuth values
        p = beta.shape[0]

        # initialize
        G = np.zeros((Ntargets,p), dtype=np.float32)
        V = np.zeros((Ntargets,Ntargets), dtype=np.float32)
        M = np.zeros((Ntargets,Ntargets), dtype=np.float32)

        for i in range(Ntargets):
            G[i,:] = weights[i]*np.array([np.cos(X[i] + delta[i]), np.sin(X[i] + delta[i])])
            V[i,i] = weights[i]*(-beta[0]*np.sin(X[i] + delta[i]) + beta[1]*np.cos(X[i] + delta[i]))

            ## (ODR-1987 Prop. 2.1)
            w =  V[i,i]**2 / E[i,i]
            M[i,i] = np.sqrt(1/(1+w));

        return G, V, M

    def getJacobian3D(self, X, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special properites
        V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix
        """

        Ntargets = X.shape[0]   # X is a column vector of azimuth values
        p = beta.shape[0]
        m = int(delta.shape[0] / Ntargets)

        theta = X[:,0]
        phi   = X[:,1]

        ## "un-interleave" delta vector into (Ntargets x m) matrix
        delta = delta.reshape((Ntargets,m))
        delta_theta = delta[:,0]
        delta_phi   = delta[:,1]

        ## defined to simplify the following calculations
        x1 = theta + delta_theta
        x2 = phi + delta_phi

        # initialize
        G = np.zeros((Ntargets,p), dtype=np.float32)
        V = np.zeros((Ntargets,Ntargets*m), dtype=np.float32)
        M = np.zeros((Ntargets,Ntargets), dtype=np.float32)

        for i in range(Ntargets):
            G[i,:] = weights[i] * np.array([np.cos(x1[i])*np.cos(x2[i]), \
                                            np.sin(x1[i])*np.cos(x2[i]), \
                                            np.sin(x2[i])])

            # V[i,2*i:2*i+2] = weights[i] * np.array([
            #     -beta[0]*np.sin(x1[i])*np.cos(x2[i]) + beta[1]*np.cos(x1[i])*np.cos(x2[i]), \
            #     -beta[0]*np.cos(x1[i])*np.sin(x2[i]) - beta[1]*np.sin(x1[i])*np.sin(x2[i]) + \
            #      beta[2]*np.cos(x2[i])])

            V[i,2*i] = weights[i]*(-beta[0]*np.sin(x1[i])*np.cos(x2[i]) + \
                                      beta[1]*np.cos(x1[i])*np.cos(x2[i]))

            V[i,2*i+1] = weights[i]*(-beta[0]*np.cos(x1[i])*np.sin(x2[i]) - \
                                    beta[1]*np.sin(x1[i])*np.sin(x2[i]) + \
                                    beta[2]*np.cos(x2[i]))

            ## (ODR-1987 Prop. 2.1)
            w =  (V[i,2*i]**2 / E[2*i,2*i]) + (V[i,2*i+1]**2 / E[2*i+1,2*i+1])
            M[i,i] = np.sqrt(1/(1+w));

        return G, V, M

    def getWeights(self):
        pass

    def getCovariance(self, Gbar, D, eps, delta, weights):
        """
        Computes the (pxp) covariance of the model parameters beta according to
        the method described in "The Computation and Use of the Asymtotic
        Covariance Matrix for Measurement Error Models", Boggs & Rogers (1989).
        """

        n = Gbar.shape[0]               # number of targets in the scan
        p = Gbar.shape[1]               # dimension of the model parameters
        m = int(delta.shape[0] / n)     # dimension of 'explanatory variable' vector

        ## form complete residual vector, g
        g = np.vstack((np.reshape(eps,(n,1)),np.reshape(delta,(n*m,1))))

        ## residual weighting matrix, Omega
        W = np.diag(np.square(weights)).astype(np.float32)
        Omega1 = np.column_stack((W, np.zeros((n,n*m), dtype=np.float32)))
        Omega2 = np.column_stack((np.zeros((n*m,n), dtype=np.float32), np.matmul(D,D)))
        Omega = np.vstack((Omega1, Omega2))

        ## compute total weighted covariance matrix of model parameters (pxp)
        self.covariance_ = ( 1/(n-p) * multi_dot([g.T,Omega,g]) ) * inv(np.matmul(Gbar.T,Gbar))

        return


def test_odr(model):
    import pylab

    ## define MLESAC parameters
    report_scores = False

    ## define ODR parameters
    converge_thres = 0.0005
    max_iter = 50
    get_covar = True

    ## init instances of MLESAC class
    base_estimator = dopplerMLESAC(model)
    mlesac = MLESAC(base_estimator,report_scores,False)
    mlesac_ols = MLESAC(base_estimator,report_scores,True,True)

    ## init instance of ODR class
    odr = OrthogonalDistanceRegression(model,converge_thres,max_iter,debug=False)

    ## outlier std deviation
    sigma_vr_outlier = 1.5

    radar_angle_bins = np.genfromtxt('../../data/1642_azimuth_bins.csv', delimiter=',')

    ## simulated 'true' platform velocity range
    min_vel = -2.5      # [m/s]
    max_vel = 2.5       # [m/s]

    ## number of simulated targets
    Ninliers = 125
    Noutliers = 35

    ## generate truth velocity vector
    velocity = (max_vel-min_vel)*np.random.random((base_estimator.sample_size,)) + min_vel

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

    ## concatrnate radar data
    radar_data = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))

    ## get MLSESAC solution
    start_time = time.time()
    mlesac.mlesac(radar_data)
    model_mlesac = mlesac.estimator_.param_vec_mlesac_
    mlesac_inliers = mlesac.inliers_
    time_mlesac = time.time() - start_time

    print("Ninliers = " + str(mlesac.inliers_.shape[0]))

    ## get MLSESAC + OLS solution
    start_time = time.time()
    mlesac_ols.mlesac(radar_data)
    model_mlesac_ols = mlesac.estimator_.param_vec_ols_
    time_mlesac_ols = time.time() - start_time

    ## concatenate inlier data for ODR regression
    odr_data = np.column_stack((radar_doppler[mlesac.inliers_], \
                                radar_azimuth[mlesac.inliers_], \
                                radar_elevation[mlesac.inliers_]))

    ## get MLESAC + ODR solution
    start_time = time.time()
    weights = (1/model.sigma_vr)*np.ones((mlesac.inliers_.shape[0],), dtype=np.float32)
    odr.odr(odr_data, model_mlesac, weights, get_covar)
    model_odr = odr.param_vec_
    odr_cov = odr.covariance_
    time_odr = time.time() - start_time

    print("\nMLESAC + ODR Velocity Profile Estimation:\n")
    print("Truth\t MLESAC\t\t MLESAC+OLS\t MLESAC+ODR")
    for i in range(base_estimator.sample_size):
        print(str.format('{0:.4f}',velocity[i]) + "\t " + str.format('{0:.4f}',model_mlesac[i]) \
              + " \t " + str.format('{0:.4f}',model_mlesac_ols[i]) \
              + " \t " + str.format('{0:.4f}',model_odr[i]))

    rmse_mlesac = np.sqrt(np.mean(np.square(velocity - model_mlesac)))
    rmse_mlesac_ols = np.sqrt(np.mean(np.square(velocity - model_mlesac_ols)))
    rmse_odr = np.sqrt(np.mean(np.square(velocity - model_odr)))
    # rmse_mlesac = np.sqrt(np.square(velocity - model_mlesac))
    # rmse_mlesac_ols = np.sqrt(np.square(velocity - model_mlesac_ols))

    print("\n\t\tRMSE [m/s]\tExec. Time [ms]")
    print("MLESAC\t\t" + str.format('{0:.4f}',rmse_mlesac) + "\t\t" + \
          str.format('{0:.2f}',1000*time_mlesac))
    print("MLESAC + OLS\t" + str.format('{0:.4f}',rmse_mlesac_ols) + "\t\t" + \
          str.format('{0:.2f}',1000*time_mlesac_ols))
    print("MLESAC + ODR\t" + str.format('{0:.4f}',rmse_odr) + "\t\t" + \
          str.format('{0:.2f}',1000*(time_mlesac+time_odr)))

def test_odr_montecarlo(model):
    pass

if __name__=='__main__':
    ## define Radar Doppler model to be used
    # model = RadarDopplerModel2D()
    model = RadarDopplerModel3D()

    test_odr(model)
