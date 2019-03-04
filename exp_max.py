"""

Functions for performing the expectation maximisation algorithm over the
observed spike-pattern rates and the natural parameters. These functions
use 'call by reference' in that, rather than returning a result, they update
the data referred to by their parameters.

---
This code fits a State-space Model of Time-varying multi-graph Ising model. It
is based on previous work on State-space Model of Time-varying Neural 
Interactions previously developed (Shimazaki et al. PLoS Comp Biol 2012; 
Donner et al. PLoS Comp Biol 2017). We acknowledge Thomas Sharp and Christian 
Donner respectively for providing codes for the inference method (from 
repository <https://github.com/tomxsharp/ssll> or 
<http://github.com/shimazaki/dynamic_corr> for Matlab), and its approximation
methods (from repository <https://github.com/christiando/ssll_lib>).

In this library, we modify the existing codes to extract the dominant 
correlation structures and their contribution to generating binary data in a
time-dependant manner. The dynamics of binary patterns are modeled with a
state-space model of an Ising type network composed of multiple undirected
graphs. For details see:
    
Jimmy Gaudreault, Arunabh Saxena and Hideaki Shimazaki. (2019). Online
Estimation of Multiple Dynamic Graphs in Pattern Sequences.  
arXiv:1901.07298. 
<https://arxiv.org/abs/1901.07298>

Copyright (C) 2019

Authors of the analysis methods: Jimmy Gaudreault (jimmy.gaudreault@polymtl.ca)
                                 Arunabh Saxena (arunabh96@gmail.com)
                                 Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)
                                 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy
import pdb

import probability
import transforms
import compute_gradient



CONVERGED = 1e-4



def compute_A(sigma_t0, sigma_t1, F):
    """
    Computes the smoothing parameter A.
    :param numpy.ndarray sigma_t0:
        Covariance of the theta distribution for timestep t.
    :param numpy.ndarray sigma_t1:
        Covariance of the theta distribution for timestep t+1.
    :param numpy.ndarray F:
        Autoregressive parameter of state transisions, of dimensions (D, D).
    :returns:
        Smoothing parameter A as a numpy.ndarray.
    """
    a = numpy.dot(sigma_t0, F.T)
    A = numpy.dot(a, numpy.linalg.inv(sigma_t1))

    return A


def e_step(emd):
    """
    Computes the expectation (a multivariate Gaussian distribution) of the
    natural parameters of observed spike patterns, given the state-transition
    hyperparameters. Firstly performs a `forward' iteration, in which the
    expectation at time t is determined from the observed patterns at time t and
    the expectation at time t-1. Secondly performs a `backward' iteration, in
    which these sequential expectation estimates are smoothed over time.
    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Compute the 'forward' filter density
    e_step_filter(emd)
    # Compute the 'backward' smooth density
    e_step_smooth(emd)


def e_step_filter(emd):
    """
    Computes the one-step-prediction density and the filter density in the
    expectation step.
    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Iterate forwards over each timestep, computing filter density
    emd.theta_f[0,:], emd.sigma_f[0,:] = emd.max_posterior(emd, 0)
    for i in range(1, emd.T):
        # Compute one-step prediction density
        emd.theta_o[i,:] = numpy.dot(emd.F, emd.theta_f[i-1,:])
        tmp = numpy.dot(emd.F, emd.sigma_f[i-1,:,:])
        emd.sigma_o[i,:,:] = numpy.dot(tmp, emd.F.T) + emd.Q
        # Get MAP estimate of filter density
        emd.theta_f[i,:], emd.sigma_f[i,:] = emd.max_posterior(emd, i)


def e_step_smooth(emd):
    """
    Computes smooth density in the expectation step.
    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Initialise the smoothed theta and sigma values
    emd.theta_s[-1,:] = emd.theta_f[-1,:]
    emd.sigma_s[-1,:,:] = emd.sigma_f[-1,:,:]
    # Iterate backwards over each timestep, computing smooth density
    for i in reversed(range(emd.T - 1)):
        # Compute the A matrix
        A = compute_A(emd.sigma_f[i,:,:], emd.sigma_o[i+1,:,:], emd.F)
        # Compute the backward-smoothed means
        tmp = numpy.dot(A, emd.theta_s[i+1,:] - emd.theta_o[i+1,:])
        emd.theta_s[i,:] = emd.theta_f[i,:] + tmp
        # Compute the backward-smoothed covariances
        tmp = numpy.dot(A, emd.sigma_s[i+1,:,:] - emd.sigma_o[i+1,:,:])
        tmp = numpy.dot(tmp, A.T)
        emd.sigma_s[i,:,:] = emd.sigma_f[i,:,:] + tmp


def m_step(emd):
    """
    Computes the optimised hyperparameters of the natural parameters of the
    expectation distributions over time. `Q' is the covariance matrix of the
    transition probability distribution. `F' is the autoregressive parameter of
    the state transitions.
    :param container.EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Update the initial mean of the one-step-prediction density
    emd.theta_o[0,:] = emd.theta_s[0,:]
    # Compute the feature matrix update
    m_step_J(emd)


def m_step_J(emd):
    """
    Computes the optimised network parameters (theta) for the original Ising
    models, given the observed data at all time.
    :param container .EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Number of samples of the whole sequence
    n_samples = 20
    # Compute the gradient
    gradient = compute_gradient.gradient_q_J(n_samples, emd)
    # Perform one iteration of the gradient ascent method
    emd.J += 1e-2 * gradient
    # Update the feature vector 
    emd.f = transforms.compute_f_LD(emd.F_tilde, emd.J)


    
