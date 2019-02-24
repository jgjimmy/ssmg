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