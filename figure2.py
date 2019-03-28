"""
Traces figure 2. "figure2_data_fit.py" must be run before this code, so that 
the necessary data is computed and saved.

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
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import os

mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.width'] = .3
mpl.rcParams['ytick.major.width'] = .3
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
fontsize = 6

# Number of pixels
N = 9

# Exact solution boolean
exact = True

# Import figure data (generated with 'figure2_data_fit.py')
directory = os.getcwd()
f = open(directory+'/Data/Data_fig2', 'rb') # Generated with gen_data_fit.py
data = pickle.load(f)
f.close()
theta_gen = data[0]
theta_fit = -data[1]
sigma = data[2]
lm_repeat = data[3]
J_list = data[4]
J_gen = data[5]

xi1 = numpy.array([[-1,1,-1,1,1,1,-1,1,-1]]).T
xi2 = numpy.array([[1,1,1,-1,1,-1,-1,1,-1]]).T

T1, D_fit = theta_fit.shape
T = len(J_list) - 1
repeat = int(T/T1)
dim = numpy.sum(range(N+1))
D = theta_gen.shape[1]

# Sample images from underlying and fitted models
import transforms
import synthesis

if exact:
    transforms.initialise(N,2)

gen_images = list()
sampled_images = list()
seed = numpy.random.seed(1)

for t in range(-1,T1+int(T1/4)-1,int(T1/4)):
    # generative image
    mixed_theta = transforms.compute_mixed_theta(theta_gen[t,:], J_gen)
    if exact:
        p = transforms.compute_p(mixed_theta.reshape(dim,1)).reshape(1,2**N)
        gen_images.append(synthesis.generate_spikes(p, 1, seed=seed))
    else:
        gen_images.append(synthesis.generate_spikes_gibbs(mixed_theta.reshape(1,dim), N, \
                                                 2, 1, seed=seed))
    # sampled images
    mixed_theta = transforms.compute_mixed_theta(theta_fit[t,:], -J_list[T-T1+t])
    if exact:
        p = transforms.compute_p(mixed_theta.reshape(dim,1)).reshape(1,2**N)
        sampled_images.append(synthesis.generate_spikes(p, 1, seed=seed))
    else:
        sampled_images.append(synthesis.generate_spikes_gibbs(mixed_theta.reshape(1,dim), N, \
                                                 2, 1, seed=seed))
# J correlation
d1 = list()
d2 = list()
a = 0
b = 0
c = 1
d = 1
for t in range(T+1):
    tmp1 = numpy.sqrt(J_list[t][:,a].T.dot(J_list[t][:,a]))
    tmp2 = numpy.sqrt(J_gen[:,b].T.dot(J_gen[:,b]))
    tmp3 = J_list[t][:,a].T.dot(J_gen[:,b])
    tmpa = tmp3/tmp1/tmp2
    d1.append(tmpa)
    tmp1 = numpy.sqrt(J_list[t][:,c].T.dot(J_list[t][:,c]))
    tmp2 = numpy.sqrt(J_gen[:,d].T.dot(J_gen[:,d]))
    tmp3 = J_list[t][:,c].T.dot(J_gen[:,d])
    tmpa = tmp3/tmp1/tmp2
    d2.append(tmpa)
    
### Figure ###
fig = plt.figure(figsize=(10,5),dpi=600)

# Generative Image 1
ax = fig.add_axes([0.05,0.825,.06,.12])
ax.imshow(xi1.reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')

# Generative Image 2
ax = fig.add_axes([0.125,0.825,.06,.12])
ax.imshow(xi2.reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')

# Thetas
ax = fig.add_axes([0.05,0.25,.5,.4])
ax.set_frame_on(False)
ax.plot(theta_fit[:,0], 'blue')
ax.fill_between(range(T1), theta_fit[:,0] - 2*numpy.sqrt(sigma[:,0]), \
                theta_fit[:,0] + 2*numpy.sqrt(sigma[:,0]), color=(194/255,211/255,223/255))
ax.plot(theta_fit[:,1], 'orange')
ax.fill_between(range(T1), theta_fit[:,1] - 2*numpy.sqrt(sigma[:,1]), \
                theta_fit[:,1] + 2*numpy.sqrt(sigma[:,1]), color=(255/255,207/255,144/255))
dashes = [1, 1, 1, 1]
ax.plot(theta_gen[:,0], 'blue', dashes=dashes)
ax.plot(theta_gen[:,1], 'orange', dashes=dashes)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ax.legend([r'$\theta^1_{fit}$',\
           r'$\theta^2_{fit}$', r'$\theta^1_{gen}$', r'$\theta^2_{gen}$'],\
          fontsize=fontsize, ncol=2, loc=4, handlelength=3)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin+0.01, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin+0.01, ymin+0.01), color='black', linewidth=1))

# Generated images
ax = fig.add_axes([0.05,0.67,.05,.1])
ax.imshow(gen_images[0].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.1625,0.67,.05,.1])
ax.imshow(gen_images[1].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.275,0.67,.05,.1])
ax.imshow(gen_images[2].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.3875,0.67,.05,.1])
ax.imshow(gen_images[3].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.5,0.67,.05,.1])
ax.imshow(gen_images[4].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')

# Sampled images
ax = fig.add_axes([0.05,0.07,.05,.1])
ax.imshow(sampled_images[0].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.1625,0.07,.05,.1])
ax.imshow(sampled_images[1].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.275,0.07,.05,.1])
ax.imshow(sampled_images[2].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.3875,0.07,.05,.1])
ax.imshow(sampled_images[3].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')
ax = fig.add_axes([0.5,0.07,.05,.1])
ax.imshow(sampled_images[4].reshape(int(N**.5),int(N**.5)), 'Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('auto')

# Marginal Log-likelihood
ax = fig.add_axes([0.625,0.58,0.35,0.38])
ax.set_frame_on(False)
ax.plot(lm_repeat, 'black')
ax.set_ylabel('Average marginal log-likelihood', fontsize=fontsize)
ax.set_xticks([])
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

# Correlation
ax = fig.add_axes([0.625,0.09,0.35,0.38])
ax.set_frame_on(False)
ax.plot(d1)
ax.plot(d2)
ax.set_ylabel('Correlation coefficient', fontsize=fontsize)
ax.set_xticklabels(range(0,225,25))
ax.set_xticks(range(0,225*T1,25*T1))
ax.set_xlabel('Number of epochs', fontsize=fontsize)
ax.legend([r'$Corr(\mathbf{J}^1_{gen}, \mathbf{J}^1_{fit})$',\
           r'$Corr(\mathbf{J}^2_{gen}, \mathbf{J}^2_{fit})$'], fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin+0.01, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin+0.01, ymin+0.01), color='black', linewidth=1))


# A, B, C
ax = fig.add_axes([0.03,0.92,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'A', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.03,0.75,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'B', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.57,0.92,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'C', fontsize=fontsize, fontweight='bold')

# Save figure
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig_2'+'.eps')
