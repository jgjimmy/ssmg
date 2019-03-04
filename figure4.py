"""
Traces figure 4. "figure4_data_fit.py" must be run before this code, so that 
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
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pickle
import networkx as nx
import itertools
import os

import transforms

# Import spike array (generated with 'extract_data_spikes.py')
n_culture = 28 # Number of culture
n_DIV = 34 # Number of days in vitro
directory = os.getcwd()
f = open(directory+'/Data/spikes_Culture'+str(n_culture)+'DIV'+str(n_DIV), 'rb')
data = pickle.load(f)
f.close()

# Import stochastic gradient results (Generated with 'figure4_data_fit.py')
import os
directory = os.getcwd()
f = open(directory+'/Data/Data_fig4_D_fit=1', 'rb') 
data1 = pickle.load(f)
f.close()
lm1 = data1[1]
f = open(directory+'/Data/Data_fig4_D_fit=2', 'rb')
data2 = pickle.load(f)
f.close()
lm2 = data2[1]
f = open(directory+'/Data/Data_fig4_D_fit=3', 'rb')
data3 = pickle.load(f)
f.close()
lm3 = data3[1]

# Extract spikes array (T,N)
spikes = data[0]

# Number of time bins
T = spikes.shape[0]
# Number of pitches
N = spikes.shape[1]

# Number of dimensions of the graphs
dim = transforms.compute_D(N,2)

# AIC using the last half of the data
TI = int(T/2+1)

AIC1 = -2*sum(lm1[-TI:]) + 2*dim*1
AIC2 = -2*sum(lm2[-TI:]) + 2*dim*2
AIC3 = -2*sum(lm3[-TI:]) + 2*dim*3

# Correlation coefficients between J at every time step and J at last time step
# D_fit =1
J = data1[2]
jj = J[-1][:,0]
tmp3 = jj.dot(jj)
corr1 = numpy.zeros((T))
for t in range(T):
    tmp1 = J[t][:,0].dot(jj)
    tmp2 = J[t][:,0].dot(J[t][:,0])
    corr1[t] = tmp1/numpy.sqrt(tmp2*tmp3)
    
# D_fit = 2
J = data2[2]
jj1 = J[-1][:,0]
jj2 = J[-1][:,1]
tmp31 = jj1.dot(jj1)
tmp32 = jj2.dot(jj2)
corr21 = numpy.zeros((T))
corr22 = numpy.zeros((T))
for t in range(T):
    tmp11 = J[t][:,0].dot(jj1)
    tmp21 = J[t][:,0].dot(J[t][:,0])
    corr21[t] = tmp11/numpy.sqrt(tmp21*tmp31)
    tmp12 = J[t][:,1].dot(jj2)
    tmp22 = J[t][:,1].dot(J[t][:,1])
    corr22[t] = tmp12/numpy.sqrt(tmp22*tmp32)
    
# D_fit = 3
J = data3[2]
jj1 = J[-1][:,0]
jj2 = J[-1][:,1]
jj3 = J[-1][:,2]
tmp31 = jj1.dot(jj1)
tmp32 = jj2.dot(jj2)
tmp33 = jj3.dot(jj3)
corr31 = numpy.zeros((T))
corr32 = numpy.zeros((T))
corr33 = numpy.zeros((T))
for t in range(T):
    tmp11 = J[t][:,0].dot(jj1)
    tmp21 = J[t][:,0].dot(J[t][:,0])
    corr31[t] = tmp11/numpy.sqrt(tmp21*tmp31)
    tmp12 = J[t][:,1].dot(jj2)
    tmp22 = J[t][:,1].dot(J[t][:,1])
    corr32[t] = tmp12/numpy.sqrt(tmp22*tmp32)
    tmp13 = J[t][:,2].dot(jj3)
    tmp23 = J[t][:,2].dot(J[t][:,2])
    corr33[t] = tmp13/numpy.sqrt(tmp23*tmp33)

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

# Spike array
ax = fig.add_axes([0.05,0.75,.4,.2])
ax.imshow(spikes.transpose(),'Greys',aspect=6000)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_ylabel('Neuron', fontsize=fontsize)
ax.set_yticks([])
ax.set_xticks([0, T/4+1, T/2+1, 3*T/4+1, T+1])
ax.set_xticklabels(['0', str(int(T/4+1)), str(int(T/2+1)), str(int(3*T/4+1)), str(T+1)])

# Fitted theta D_fit = 1
ax = fig.add_axes([0.05,0.5,.4,.175])
ax.set_title(r'$D_{fit}=1$',fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data1[0])
ax.set_xticks([])               
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

# Fitted theta D_fit = 2
ax = fig.add_axes([0.05,0.3,.4,.175])
ax.set_title(r'$D_{fit}=2$',fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data2[0][:,0],'b')
ax.plot(data2[0][:,1],'orange')
ax.set_xticks([])
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))

# Fitted theta D_fit = 3
ax = fig.add_axes([0.05,0.1,.4,.175])
ax.set_title(r'$D_{fit}=3$',fontsize=fontsize+2)
ax.set_frame_on(False)
ax.plot(data3[0][:,0],'b')
ax.plot(data3[0][:,1],'orange')
ax.plot(data3[0][:,2],'g')
ax.set_ylabel(r'$\theta_t$', fontsize=fontsize)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_xticks([0, T/4+1, T/2+1, 3*T/4+1, T+1])
ax.set_xticklabels(['0', str(int(T/4+1)), str(int(T/2+1)), str(int(3*T/4+1)), str(T+1)])
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.legend([r'$\theta^1_{fit}$',\
           r'$\theta^2_{fit}$', r'$\theta^3_{fit}$'],\
          fontsize=fontsize, ncol=3, loc=3)

# First column of J
J = data3[2][-1]
J1 = J[:,0]

ax = fig.add_axes([0.4475,0.015,0.35,0.35])

all_conns = itertools.combinations(range(N),2)
conns = numpy.array(list(all_conns))

G1 = nx.Graph()
G1.add_nodes_from(range(N))
G1.add_edges_from(conns)
pos1 = nx.circular_layout(G1, scale=0.0015)
net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=ax, node_color=J1[:N],
                                   cmap=pyplot.get_cmap('hot'), vmin=-3,vmax=3., node_size=25, linewidths=.5)
e1 = nx.draw_networkx_edges(G1, pos1, ax=ax, edge_color=J1[N:].tolist(),
                            edge_cmap=pyplot.get_cmap('seismic'),edge_vmin=-2.,edge_vmax=2., width=1)
ax.axis('off')
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

# Second column of J
J1 = J[:,1]

ax = fig.add_axes([0.5475,0.015,0.35,0.35])

G1 = nx.Graph()
G1.add_nodes_from(range(N))
G1.add_edges_from(conns)
pos1 = nx.circular_layout(G1, scale=0.0015)
net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=ax, node_color=J1[:N],
                                   cmap=pyplot.get_cmap('hot'), vmin=-3,vmax=3., node_size=25, linewidths=.5)
e1 = nx.draw_networkx_edges(G1, pos1, ax=ax, edge_color=J1[N:].tolist(),
                            edge_cmap=pyplot.get_cmap('seismic'),edge_vmin=-2.,edge_vmax=2., width=1)
ax.axis('off')
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

# Third column of J
J1 = J[:,2]

ax = fig.add_axes([0.6475,0.015,0.35,0.35])

G1 = nx.Graph()
G1.add_nodes_from(range(N))
G1.add_edges_from(conns)
pos1 = nx.circular_layout(G1, scale=0.0015)
net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=ax, node_color=J1[:N],
                                   cmap=pyplot.get_cmap('hot'), vmin=-3,vmax=3., node_size=25, linewidths=.5)
e1 = nx.draw_networkx_edges(G1, pos1, ax=ax, edge_color=J1[N:].tolist(),
                            edge_cmap=pyplot.get_cmap('seismic'),edge_vmin=-2.,edge_vmax=2., width=1)
ax.axis('off')
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

# Colorbars
cbar_ax = fig.add_axes([0.55, 0.135, 0.01, 0.1])
cbar = fig.colorbar(net_nodes, cax=cbar_ax, orientation='vertical')
cbar_ax.yaxis.set_ticks_position('left')
cbar.outline.set_linewidth(.5)
cbar.set_ticks([-3,0,3])
cbar_ax.set_title('$h_{i}$', fontsize=fontsize)
cbar_ax = fig.add_axes([0.885, 0.135, 0.01, 0.1])
cbar = fig.colorbar(e1, cax=cbar_ax, orientation='vertical')
cbar.outline.set_linewidth(.5)
cbar.set_ticks([-2,0.,2])
cbar_ax.set_title('$j_{ij}$', fontsize=fontsize)

# AIC
ax = fig.add_axes([0.625,0.4,.2,.2])
rects = ax.bar([1,2,3],[AIC1,AIC2,AIC3],width=0.3)
ax.xaxis.tick_bottom()
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks([1,2,3])
ax.set_xticklabels((r'$D_{fit}=1$',r'$D_{fit}=2$',r'$D_{fit}=3$'))
ax.set_frame_on(False)
ax.set_ylabel('AIC', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            '%d' % int(height),
            ha='center', va='bottom',fontsize=fontsize)
    
# Correlations
ax = fig.add_axes([0.575, 0.7, 0.3, 0.225])
ax.set_frame_on(False)
ax.plot(corr1,'b')
ax.plot(corr21,'orange')
ax.plot(corr31,'g')
ax.plot(corr22,'orange')
ax.plot(corr32,'g')
ax.plot(corr33,'g')
ax.set_ylabel('Correlation coefficient', fontsize=fontsize)
ax.set_xlabel('Time step', fontsize=fontsize)
ax.set_xticks([0, (T)/4+1, (T)/2+1, 3*(T)/4+1, (T)+1])
ax.set_xticklabels(['0', str(int((T)/4+1)), str(int((T)/2+1)), str(int(3*(T)/4+1)), str((T)+1)])
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.legend([r'$D_{fit}=1$',r'$D_{fit}=2$', r'$D_{fit}=3$'],\
          fontsize=fontsize, ncol=3, loc=4)

# A, B, C, D, E
ax = fig.add_axes([0.01,0.92,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'A', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.01,0.695,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'B', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.5,0.92,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'C', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.5,0.635,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'D', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.5,0.325,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'E', fontsize=fontsize, fontweight='bold')

# Save figure
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig_4'+'.eps')                                     
