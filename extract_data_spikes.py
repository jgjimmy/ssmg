"""
Extracts the data from MATLAB files to pickle Python files.

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

import scipy.io
import pickle
import numpy
import os

# Load the data from the Matlab file
directory = os.getcwd()
n_culture = 28 # Number of culture
n_DIV = 34 # Number of days in vitro
mat1 = scipy.io.loadmat(directory+'/Neural_Data/Culture'+str(n_culture)+'DIV'+str(n_DIV)+'.mat')

# Total number of neurons
N = numpy.size(mat1['data']['spikes'][0][0])

# Number of neurons retained for analysis
n = 12;

# Number of spikes for each neuron
nspikes = numpy.zeros((N))

for i in range(N):
    nspikes[i] =  numpy.size(mat1['data']['spikes'][0][0][i][0][0]) 

# Choose the neurons with the highest firing rate    
neurons = numpy.argsort(nspikes)[-n:]

# Create file
events = list()

for i in neurons:
    events.append(mat1['data']['spikes'][0][0][i][0][0])

# Number of time bins
T = int(mat1['data']['nbins'][0][0][0][0]/200) # Time bins of 10ms

# Spike array
spikes = numpy.zeros((T,n))

for i in range(n):
    nspikes = len(events[i])
    for j in range(nspikes):
        t = int(events[i][j]/200)
        spikes[t,i] = 1

# Save directory
directory = os.getcwd()
if not os.path.exists(directory+'/Data/'):
    os.makedirs(directory+'/Data/')

# Save the spike array
f = open(directory+'/Data/spikes_Culture'+str(n_culture)+'DIV'+str(n_DIV), 'wb')
pickle.dump((spikes,neurons) ,f)
f.close()


