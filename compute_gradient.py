"""
Computes the gradient necessary to update the J matrix in the M step of the EM
algorithm.

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

import transforms
import exp_max
import bethe_approximation

def gradient_q_J(n_samples, emd):
    """
    Computes the gradient of the q function with respect to the
    network parameters of the Ising models of the main fetures.
    :param container .EMData emd:
        All data pertaining to the EM algorithm.
    :param int n_samples:
        Number of samples to take to approximate the expectation inside the
        equation for the gradient.
    """
    # Initialise the gradient
    gradient = numpy.zeros((emd.dim, emd.D))
    for n in range(n_samples):
        gradient += gradient_1sample(n, emd)
    gradient = gradient/n_samples
        
    return gradient 



def gradient_1sample(n_samples, emd):
    """
    Computes one sample of the expectation inside the equation for the
    gradient of the q function with respect to the network parameters of
    the Ising models of the main fetures.
    :param container .EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Take 1 sample of theta for all time steps
    generated_theta = sample_theta(emd)
    # Compute 1 sample
    sample_value = 0
    for t in range(emd.T):
        theta_mixed = transforms.compute_mixed_theta(generated_theta[t,:], \
                                                     emd.J)
        if emd.exact:
            p = transforms.compute_p(theta_mixed)
            eta_tilde = transforms.compute_eta(p)
        else:
            eta_tilde = bethe_approximation.compute_eta_hybrid(theta_mixed, emd.N)
        a = emd.F_tilde[t,:] - eta_tilde
        # Summation for all time bins
        sample_value += emd.R * numpy.dot(a[:,numpy.newaxis], \
                                          generated_theta[t,numpy.newaxis,:])

    return sample_value



def sample_theta(emd):
    """
    Samples the sequence of theta over all time step by assuming a Gaussian
    distribution.
    :param container .EMData emd:
        All data pertaining to the EM algorithm.
    """
    # Initialise the generated theta
    generated_theta = numpy.zeros((emd.T, emd.D))
    # First time bin
    generated_theta[0,:] = numpy.random.multivariate_normal(emd.theta_s[0,:],\
                                                            emd.sigma_s[0,:,:],\
                                                            1)
    # Iterate over all time steps
    for t in range(1, emd.T):
        A = exp_max.compute_A(emd.sigma_f[t-1,:,:], emd.sigma_o[t,:,:], emd.F)
        lag_one_covariance = numpy.dot(A, emd.sigma_s[t,:])
        mean = emd.theta_s[t,:] + lag_one_covariance.T \
               .dot(numpy.linalg.inv(emd.sigma_s[t-1,:,:])) \
               .dot(generated_theta[t-1,:] - emd.theta_s[t-1,:])
        covariance = emd.sigma_s[t,:] - lag_one_covariance.T \
                     .dot(numpy.linalg.inv(emd.sigma_s[t-1,:,:])) \
                     .dot(lag_one_covariance)
        generated_theta[t,:] = numpy.random.multivariate_normal(mean, \
                                                                covariance,\
                                                                1)        

    return generated_theta


        
