"""

Extracts the data from MATLAB files to pickle Python files.

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


