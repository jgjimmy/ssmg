import matplotlib.pyplot as plt
import pickle
import os

import transforms

# Import stochastic gradient results (generated with 'figure3_data_fit.py')
directory = os.getcwd()
f = open(directory+'/Data/Data_fig3_D_fit=1', 'rb') # Generated with gen_data_fit.py
data1 = pickle.load(f)
f.close()
lm1 = data1[3]
f = open(directory+'/Data/Data_fig3_D_fit=2', 'rb')
data2 = pickle.load(f)
f.close()
lm2 = data2[3]
f = open(directory+'/Data/Data_fig3_D_fit=3', 'rb')
data3 = pickle.load(f)
f.close()
lm3 = data3[3]
f = open(directory+'/Data/Data_fig3_D_fit=4', 'rb')
data4 = pickle.load(f)
f.close()
lm4 = data4[3]

# Number of time bins
T = data1[0].shape[0]
# Number of neurons per graph
N = data1[-1][0].shape[2]
# Number of dimensions
dim = transforms.compute_D(N,2)

# Compute AIC for last repetition :
# -2 * marginal log-likelihood + nb hyperparameters * 2
T1 = 1500
aic1 = -2 * T1 * lm1[-1] + dim * 1 * 2
aic2 = -2 * T1 * lm2[-1] + dim * 2 * 2
aic3 = -2 * T1 * lm3[-1] + dim * 3 * 2
aic4 = -2 * T1 * lm4[-1] + dim * 4 * 2

### Figure ###
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.width'] = .3
mpl.rcParams['ytick.major.width'] = .3
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
fontsize = 6

fig = plt.figure(figsize=(10,5),dpi=600)

# Generative thetas
ax = fig.add_axes([0.2,0.65,.6,.25])
ax.set_title('Generative model', fontsize=fontsize+2)
ax.plot(data2[0][-T:])
ax.set_frame_on(False)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ax.set_xticks([0, T/4, T/2, 3*T/4, T])
ax.set_xticklabels(['0', str(int(T/4)), str(int(T/2)), str(int(3*T/4)), str(T)])
ax.legend([r'$\theta^1_{gen}$',\
           r'$\theta^2_{gen}$'], fontsize=fontsize, ncol=2, loc=4)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))

# Fitted theta D_fit = 1
ax = fig.add_axes([0.05,0.35,.4,.2])
ax.set_title(r'$D_{fit}=1$, AIC=%1.0f' %aic1, fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data1[1])
ax.set_xticks([])
ax.set_yticks([-1,-0.5,0,0.5,1])               
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

# Fitted theta D_fit = 2
ax = fig.add_axes([0.55,0.35,.4,.2])
ax.set_title(r'$D_{fit}=2$, AIC=%1.0f' %aic2, fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data2[1])
ax.set_xticks([])
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

# Fitted theta D_fit = 3
ax = fig.add_axes([0.05,0.1,.4,.2])
ax.set_title(r'$D_{fit}=3$, AIC=%1.0f' %aic3, fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data3[1])
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_xticks([0, T/4, T/2, 3*T/4, T])
ax.set_xticklabels(['0', str(int(T/4)), str(int(T/2)), str(int(3*T/4)), str(T)])
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))

# Fitted theta D_fit = 4
ax = fig.add_axes([0.55,0.1,.4,.2])
ax.set_title(r'$D_{fit}=4$, AIC=%1.0f' %aic4, fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data4[1])
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_xticks([0, T/4, T/2, 3*T/4, T])
ax.set_xticklabels(['0', str(int(T/4)), str(int(T/2)), str(int(3*T/4)), str(T)])
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.legend([r'$\theta^1_{fit}$',\
           r'$\theta^2_{fit}$', r'$\theta^3_{fit}$', r'$\theta^4_{fit}$'],\
          fontsize=fontsize, ncol=2, loc=4)

# A, B, C
ax = fig.add_axes([0.16,0.92,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'A', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.01,0.53,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'B', fontsize=fontsize, fontweight='bold')

# Save figure
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig_3'+'.eps')                                     
