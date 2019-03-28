"""
Fits the data necessary to trace figure 3. "gen_spikes_figure3.py" must be run
before this code, so that the spike array on which to fit the model is created
and saved. To trace figure 3, this code need to be run 4 times, by setting 
parameter "D_fit" to 1, 2, 3 and 4 (line 60).

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

##### GENERATIVE MODEL #####
# Exact solution boolean
exact = True

# Number of graphs in the fitted models
D_fit = 2

# Precision
lmbda = 1000

# Random seed
seed = numpy.random.seed(0)

# Initial one-step predictor (hyperparameters mu and sigma)
theta_o = numpy.zeros((1,D_fit))
sigma_o = 0.1 * numpy.eye(D_fit)

# Load spikes list (generated with 'gen_spikes_figure3.py')
directory = os.getcwd()
f = open(directory+'/Data/spikes_fig3','rb')
data_tmp = pickle.load(f)
f.close()
spikes_list = data_tmp[0]
J_gen = data_tmp[1]
theta_gen = data_tmp[2]

# Number of time bins in one epoch
T1 = spikes_list[0].shape[0]

# Number of neurons in the graphs
N = spikes_list[0].shape[2]

# Number of dimensions of the graphs
dim = transforms.compute_D(N, 2)

# Number of epochs
rep = len(spikes_list)

# Total number of time bins
T = T1 * rep

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

print('D_fit='+str(D_fit))
for r in range(rep):
    print(str(r+1)+'/'+str(rep)+' epochs')
    for t in range(T1):
        # Get the spike array
        spikes = spikes_list[r][t,numpy.newaxis,:,:]
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
f = open(directory+'/Data/Data_fig3_D_fit='+str(D_fit), 'wb')
pickle.dump(data, f)
f.close()