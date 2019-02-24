"""
Master file of the State-Space Analysis of Spike Correlations.
---
State-Space Analysis of Spike Correlations (Shimazaki et al. PLoS Comp Bio 2012)
Copyright (C) 2014  Thomas Sharp (thomas.sharp@riken.jp)
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

import container
import exp_max
import probability
import max_posterior
import synthesis
import transforms



def run(spikes, D, map_function='nr',  window=1, lmbda=100, max_iter=1, \
        J=None, theta_o=0, sigma_o=0.1, exact=True):
    """
    Master-function of the State-Space Analysis of Spike Correlation package.
    Uses the expectation-maximisation algorithm to find the probability
    distributions of natural parameters of spike-train interactions over time.
    Calls slave functions to perform the expectation and maximisation steps
    repeatedly until the data likelihood reaches an asymptotic value.
    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :param int D
        Number of graphs to consider in the fitting of the model
    :param string map_function:
        Name of the function to use for maximum a-posterior estimation of the
        natural parameters at each timestep. Refer to max_posterior.py.
    :param float lmdbda:
        Coefficient on the identity matrix of the initial state-transition
        covariance matrix.
    :param int max_iter:
        Maximum number of iterations for which to run the EM algorithm.
    :param numpy.ndarray J:
        First estimate of the graphs matrix
    :param boolean exact
        Whether to use exact solution (True) or Bethe approximation (False)
        for the computation of eta, psi and fisher info
    :returns:
        Results encapsulated in a container.EMData object, containing the
        smoothed posterior probability distributions of the natural parameters
        of the spike-train interactions at each timestep, conditional upon the
        given spikes.
    """
    # Ensure NaNs are caught
    numpy.seterr(invalid='raise', under='raise')
    # Order of interactions considered in the graphs
    order_ising = 2
    # Initialise the EM-data container
    map_func = max_posterior.functions[map_function]
    emd = container.EMData(spikes, window, D, map_func, lmbda, J,\
                           theta_o, sigma_o, exact)
    # Initialise the coordinate-transform maps
    if exact:
        transforms.initialise(emd.N, order_ising)
    # Set up loop guards for the EM algorithm
    lmp = -numpy.inf
    lmc = probability.log_marginal(emd)
    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > exp_max.CONVERGED):
        # Perform EM
        exp_max.e_step(emd)
        exp_max.m_step(emd)
        # Update previous and current log marginal values
        lmp = lmc
        lmc = probability.log_marginal(emd)
        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = abs(1 - lmp / lmc)

    return emd
