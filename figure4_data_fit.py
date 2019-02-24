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