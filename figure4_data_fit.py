"""
Fits the data necessary to trace figure 4. "extract_data_spikes.py" must be run
before this code, so that the spike array on which to fit the model is 
extracted and saved. To trace figure 4, this code need to be run 3 times, by 
setting parameter "D_fit" to 1, 2 and 3 (line 56).

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
import pickle
import os

import transforms
import __init__
import probability

# Number of graphs of the fitted model
D_fit = 2

# Import spike array (generated with 'extract_data_spikes.py')
n_culture = 28 # Number of culture
n_DIV = 34 # Number of days in vitro

directory = os.getcwd()
f = open(directory+'/Data/spikes_Culture'+str(n_culture)+'DIV'+str(n_DIV), 'rb')
data = pickle.load(f)
f.close()

# Extract spikes array (T,N)
spikes = data[0]

# Add an axis for trials to work with existing code
spikes = spikes[:,numpy.newaxis,:]

# Number of time bins (5ms)
T = numpy.size(spikes,0)

# Number of neurons
N = numpy.size(spikes,2)

# Precision
lmbda = 10000

# Exact solution boolean
exact = True

# Number of dimensions
dim = transforms.compute_D(N, 2)

# First estimate of the fitted J Matrix
J = numpy.random.normal(0, 1, (dim,D_fit))

# Initial one-step predictor (hyperparameters mu and sigma)
theta_o = numpy.zeros((1,D_fit))
sigma_o = 0.1 * numpy.eye(D_fit)

# Initialise theta, sigma and marginal log likelihood
theta_fit = numpy.zeros((T,D_fit))
lm = numpy.zeros(T)

# Initialise a list of J estimates
J_list = list()
J_list.append(numpy.copy(J))

print('D_fit = '+str(D_fit))
for t in range(T):
    # Fit to update J
    emd = __init__.run(spikes[t,numpy.newaxis], D_fit, lmbda=lmbda, J=J, \
                       theta_o=theta_o, sigma_o=sigma_o, exact=exact)
    # Retrieve and normalize J (unit variance)
    J = emd.J / (numpy.var(emd.J, axis=0)**0.5)
    # Save J
    J_list.append(numpy.copy(J))
    # Update one-step predictors
    theta_o = numpy.dot(emd.F, emd.theta_f[0,:])
    tmp = numpy.dot(emd.F, emd.sigma_f[0,:,:])
    sigma_o = numpy.dot(tmp, emd.F.T) + emd.Q
    # Retrieve fitted weights and their variance
    theta_fit[t,:] = emd.theta_s[0,:]
    # Compute the current marginal log-likelihood
    lm[t] = probability.log_marginal(emd)
    print('step '+str(t)+'/ '+str(T))

# Save results
data = (theta_fit, lm, J_list)
f = open(directory+'/Data/Data_fig4_D_fit='+str(D_fit), 'wb')
pickle.dump(data, f)
f.close()