"""
Classes for encapsulating data used in the expectation maximisation algorithm.

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
import pdb
import transforms


class EMData:
    """
    Contains all of the data used by the EM algorithm, purely for convenience.
    Takes spike trains as an input and computes the observable spike
    (co)incidences (patterns). Initialises the means and covariances of the
    filtered- smoothed- and one-step-prediction natural-parameter distributions.
    Initialises the autoregressive and state-transition hyperparameters.
    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :param int D
        Number of graphs to consider in the fitting of the model
    :param function map_function:
        A function from max_posterior.py that returns an estimate of the
        posterior distribution of natural parameters for a given timestep.
    :param float lmdbda:
        Coefficient on the identity matrix of the initial state-transition
        covariance matrix.
    :param object misc:
        Any miscelaneous data, of any form, about the analysis that may be
        required for post-processing and plotting later.
    :param boolean exact
        Whether to use exact solution (True) or Bethe approximation and CCCP
        (False) for the computation of eta, psi and fisher info.
    :ivar numpy.ndarray spikes:
        Reference to the input spikes.
    :ivar int window:
        Copy of the `window' parameter.
    :ivar function max_posterior:
        A function from max_posterior.py that returns an estimate of the
        posterior distribution of natural parameters for a given timestep.
    :ivar int T:
        Number of timestep in the pattern-counts; should equal the length of the
        spike trains divided by the window.
    :ivar int R:
        Number of trials in the `spikes' input.
    :ivar int N:
        Number of cells in the `spikes' input.
    :ivar numpy.ndarray F_tilde:
        Mean rates of each spike pattern at each timestep, in a 2D array of
        dimesions (T, D).
    :ivar numpy.ndarray theta_o:
        One-step-prediction density mean. Data at theta_o[0] describes the
        probability of the initial state.
    :ivar numpy.ndarray theta_f:
        Filtered density mean.
    :ivar numpy.ndarray theta_s:
        Smoothed density mean.
    :ivar numpy.ndarray sigma_o:
        One-step-prediction density covariance.
    :ivar numpy.ndarray sigma_f:
        Filtered density covariance.
    :ivar numpy.ndarray sigma_s:
        Smoothed density covariance.
    :ivar numpy.ndarray F:
        Autoregressive parameter of state transitions.
    :ivar numpy.ndarray Q:
        Covariance matrix of state-transition probability.
    :ivar int iterations:
        Number of EM iterations for which the algorithm ran.
    :ivar float convergence:
        Ratio between previous and current log-marginal prob. on last iteration.
    :ivar numpy.ndarray J
        Graphs Matrix.
    :ivar numpy.ndarray f:
        Feature vector of the multi-graph Ising model at each timestep,
        in a 2D array of dimensions (dim, T).
    :ivar int dim
        Number of dimensions of the original Ising model.
    :ivar boolean exact
        Copy of the 'exact' parameter.
    """
    def __init__(self, spikes, window, D, map_function, lmbda, J,\
                 theta_o, sigma_o, exact):
        # Order of interactions considered in the graphs
        order_ising = 2
        # Exact boolean
        self.exact = exact
        # Record the input parameters
        self.spikes, self.window = spikes, window
        self.max_posterior = map_function
        # Count the number of original static Ising models
        self.D = D
        # Compute the `sample' spike-train interactions from the input spikes
        self.F_tilde = transforms.compute_y(self.spikes, order_ising, self.window)
        # Count timesteps, trials, cells
        T, self.R, self.N = self.spikes.shape
        self.T = self.F_tilde.shape[0]
        assert self.T == T / window
        # Dimension of the graphs
        self.dim = transforms.compute_D(self.N, order_ising)
        # Initialisation of the graphs
        if J is not None:
            self.J = J
        else:
            self.J = numpy.random.normal(0, 1, (self.dim, D))
        # Compute the feature vector for all timesteps
        self.f = transforms.compute_f_LD(self.F_tilde, self.J)
        # Initialise one-step-prediction- filtered- smoothed-density means
        self.theta_o = numpy.ones((self.T,self.D)) * theta_o
        self.theta_f = numpy.zeros((self.T,self.D))
        self.theta_s = numpy.zeros((self.T,self.D))
        # Initialise covariances of the same (an I-matrix for each timestep)
        I = [numpy.identity(self.D) for i in range(self.T)]
        I = numpy.vstack(I).reshape((self.T,self.D,self.D))
        self.sigma_o = sigma_o * I
        self.sigma_f = .1 * I
        self.sigma_s = .1 * I
        del I
        # Intialise autoregressive and transition probability hyperparameters
        self.F = numpy.identity(self.D)
        self.Q = 1/lmbda * numpy.identity(self.D)
        # Metadata about EM algorithm execution
        self.iterations, self.convergence = 0, numpy.inf
