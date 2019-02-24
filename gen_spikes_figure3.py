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