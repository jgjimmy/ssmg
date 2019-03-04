"""
Generates the spike array necessary to fit the data that is shown on figure 3.

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
import os
import pickle

import transforms
import synthesis

##### GENERATIVE MODEL #####

# Number of neurons in the First Ising Models
N = 9

# Exact solution boolean
exact = True

# Number of graphs in the generative model
D = 2

# Random seed
seed = numpy.random.seed(0)

# p_map and e_map (Initialise)
if exact:
    transforms.initialise(N, 2)

# Number of dimensions of the graphs
dim = transforms.compute_D(N, 2)

# Number of time bins in one epoch
T1 = 1500

# Number of epochs
rep = 200

# Total number of time bins
T = T1 * rep

# Generative weights
theta_gen = numpy.zeros((T1, D))
frequency = numpy.array([1,3])
amplitude = numpy.array([0.5,-0.5])
baseline = numpy.array([0.5,0.5])
for t in range(T1):
    theta_gen[t,:] = amplitude*numpy.sin(2*numpy.pi*frequency*t/T1) + baseline

# Random J_gen
J_gen = numpy.random.normal(0, 1, (dim,D))

# Mixed theta
mixed_theta = transforms.compute_mixed_theta(theta_gen, J_gen) # (T1, dim)

# Probability vector
if exact:
    p = numpy.zeros((T1, 2**N))
    for t in range(T1):
        p[t,:] = transforms.compute_p(mixed_theta[t,:])

# Spikes list
spikes_list = list()

for r in range(rep):
    print(str(r+1)+'/'+str(rep)+' epochs')
    spikes = numpy.zeros((T1,1,N))
    for t in range(T1):
        # Get the spike array
        if exact:
            spikes[t,0,:] = synthesis.generate_spikes(p[t,numpy.newaxis,:], 1, seed=seed)
        else:
            spikes[t,0,:] = synthesis.generate_spikes_gibbs(mixed_theta[t,:].reshape(1,dim), N, \
                                                     2, 1)
    spikes_list.append(spikes)

# Save directory
directory = os.getcwd()
if not os.path.exists(directory+'/Data/'):
    os.makedirs(directory+'/Data/')
    

# Save results
f = open(directory+'/Data/spikes_fig3', 'wb')
pickle.dump((spikes_list,J_gen,theta_gen), f)
f.close()