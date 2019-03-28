"""
Fits the data necessary to trace figure 2.

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
Presented at IJCNN 2019. arXiv:1901.07298. 
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
import os
import pickle

import transforms
import synthesis
import __init__
import probability

##### GENERATIVE MODEL #####

# Number of neurons in the graphs
N = 9

# Exact solution boolean
exact = True

# Number of dimensions of the generative model
D = 2

# Number of graphs in the fitted model
D_fit = 2

# Precision
lmbda = 1000

# Random seed
seed = numpy.random.seed(0)

# p_map and e_map (Initialise)
if exact:
    transforms.initialise(N, 2)

# Number of dimensions of the graphs
dim = transforms.compute_D(N, 2)

# Generative J matrix
J_gen = numpy.zeros((dim,D_fit))
x1 = numpy.array([-1,1,-1,1,1,1,-1,1,-1])
x2 = numpy.array([1,1,1,-1,1,-1,-1,1,-1])
a1 = numpy.outer(x1,x1)
a2 = numpy.outer(x2,x2)
J_gen[:N,0] = numpy.diag(a1)
J_gen[:N,1] = numpy.diag(a2)
J_gen[N:,0] = a1[numpy.triu_indices(N, k=1)]
J_gen[N:,1] = a2[numpy.triu_indices(N, k=1)]
J_gen = J_gen/(numpy.var(J_gen, axis=0)**0.5)

# Number of time bins in one epoch
T1 = 1500

# Number of epochs
rep = 200

# Total number of time bins
T = T1 * rep

# Generative weights
theta_gen = numpy.zeros((T1, D))
frequency = numpy.array([1,1])
amplitude = numpy.array([0.5,-0.5])
baseline = numpy.array([0.5,0.5])
for t in range(T1):
    theta_gen[t,:] = amplitude*numpy.sin(2*numpy.pi*frequency*t/T1) + baseline

# Mixed theta
mixed_theta = transforms.compute_mixed_theta(theta_gen, J_gen) # (T1, dim)

# Probability vector
if exact:
    p = numpy.zeros((T1, 2**N))
    for t in range(T1):
        p[t,:] = transforms.compute_p(mixed_theta[t,:])

##### Fitted model #####
    
# First estimate of the fitted J Matrix
J = numpy.random.normal(0, 1, (dim,D_fit))

# Mean and covariance of weights at the last fitted epoch
theta_fit = numpy.zeros((T1,D_fit))
sigma = numpy.zeros((T1, D_fit))
# Marginal log-likelihood at the end of every epoch
lm = numpy.zeros((T1))
lm_rep = list()
# J matrix at every time step
J_list = list()
J_list.append(J)

# Initial one-step predictor (hyperparameters mu and sigma)
theta_o = numpy.zeros((1,D_fit))
sigma_o = 0.1 * numpy.eye(D_fit)

# Spikes list
spikes_list = list()

for r in range(rep):
    print(str(r+1)+'/'+str(rep)+' epochs')
    for t in range(T1):
        # Get the spike array
        if exact:
            spikes = synthesis.generate_spikes(p[t,numpy.newaxis,:], 1, seed=seed)
        else:
            spikes = synthesis.generate_spikes_gibbs(mixed_theta[t,:].reshape(1,dim), N, \
                                                     2, 1)
        # Fit to update J
        emd = __init__.run(spikes, D_fit, lmbda=lmbda, J=J, \
                       theta_o=theta_o, sigma_o=sigma_o, exact=exact)
        # Retrieve updated J 
        J = emd.J
        # Scale columns of J
        J = J / (numpy.var(J, axis=0)**0.5)
        # Save J
        J_list.append(J)
        # Update one-step predictors
        theta_o = numpy.dot(emd.F, emd.theta_f[0,:])
        tmp = numpy.dot(emd.F, emd.sigma_f[0,:,:])
        sigma_o = numpy.dot(tmp, emd.F.T) + emd.Q
        # Retrieve fitted weights and their variance
        theta_fit[t,:] = emd.theta_s[0,:]
        sigma[t,:] = numpy.diag(emd.sigma_s[0])
        # Compute the current marginal log-likelihood
        lm[t] = probability.log_marginal(emd)
        # Record the spike train
        spikes_list.append(spikes)
    lm_rep.append(lm.mean())

# Save directory
directory = os.getcwd()
if not os.path.exists(directory+'/Data/'):
    os.makedirs(directory+'/Data/')

# Save results
data = (theta_gen, theta_fit, sigma, lm_rep, J_list, J_gen, spikes_list)
f = open(directory+'/Data/Data_fig2', 'wb')
pickle.dump(data, f)
f.close()
