"""
Functions for computing coordinate transformations, derivatives and
sample means of parameters and data.

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
import itertools
import numpy
import pdb
import scipy.misc



# Matrix to map from theta to probability
p_map = None
# Matrix to map from probability to eta
eta_map = None



def compute_D(N, O):
    """
    Computes the number of natural parameters for the given number of cells and
    order of interactions.
    :param int N:
        Total cells in the spike data.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :returns:
        Number of natural parameters for the spike-pattern interactions.
    """
    D = numpy.sum([scipy.misc.comb(N, k, exact=1) for k in range(1, O + 1)])

    return D


def compute_eta(p):
    """
    Computes the expected values, eta, of spike patterns
        n_0,1 = p(0,1) + p(1,1) # for example
    from the supplied probability mass.
    :param numpy.ndarray p:
        Probability mass of spike patterns.
    :returns:
        Expected values of spike patterns (eta) as a numpy.ndarray.
    """
    global eta_map

    eta = numpy.dot(eta_map, p)

    return eta


def compute_fisher_info(p, eta):
    """
    Computes the Fisher-information matrix of the expected values, eta, of spike
    patterns for the purposes of Newton-Raphson gradient-ascent and
    computation of the marginal probability distribution. For example, for two
    neurons:
        H = [n1 - n1^2,      n12 - n1 * n2,  n12 - n1 * n12,
             n12 - n1 * n2,  n2 - n2^2,      n12 - n2 * n12,
             n12 - n1 * n12, n12 - n2 * n12, n12 - n12^2]
    :param numpy.ndarray p:
        Probability mass of spike patterns.
    :param numpy.ndarray eta:
        Expected values of spike patterns.
    :returns:
        Fisher-information matrix as a numpy.ndarray.
    """
    global p_map, eta_map

    # Stack columns of p for next step
    p_stack = numpy.repeat(p, eta.size).reshape(p.size, eta.size)
    # Compute Fisher matrix
    fisher = numpy.dot(eta_map, p_stack * p_map) - numpy.outer(eta, eta)

    return fisher


def compute_p(theta):
    """
    Computes the probability distribution of spike patterns, for example
        p(x1,x2) =     e^(t1x1)e^(t2x2)e^(t12x1x2)
                   -----------------------------------
                   1 + e^(t1) + e^(t2) + e^(t1+t2+t12)
    from the supplied natural parameters.
    :param numpy.ndarray theta:
        Natural `theta' parameters: t1, t2, ..., t12, ...
    :returns:
        Probability mass as a numpy.ndarray.
    """
    global p_map

    # Compute log probabilities
    log_p = numpy.dot(p_map, theta)
    # Take exponential and normalise
    p = numpy.exp(log_p)
    p_tmp = p / numpy.sum(p)

    return p_tmp


def compute_psi(theta):
    """
    Computes the normalisation parameter, psi, for the log-linear probability
    mass function of spike patterns. For example, for two neurons
        psi(theta) = 1 + e^(t1) + e^(t2) + e^(t1+t2+t12)
    :param numpy.ndarray theta:
        Natural `theta' parameters: t1, t2, ..., t12, ...
    :returns:
        Normalisation parameter, psi, of the log linear model as a float.
    """
    global p_map

    # Take coincident-pattern subsets of theta
    tmp = numpy.dot(p_map, theta)
    # Take the sum of the exponentials and take the log
    tmp = numpy.sum(numpy.exp(tmp))
    psi = numpy.log(tmp)

    return psi


def compute_y(spikes, order, window):
    """
    Computes the empirical mean rate of each spike pattern across trials for
    each timestep up to `order', for example
        y_12,t = 1 / N * \sigma^{L} X1_l,t * X2_l,t
    is a second-order pattern where t is the timestep and l is the trial.
    :param numpy.ndarray spikes:
        Binary matrix with dimensions (time, runs, cells), in which a `1' in
        location (t, r, c) denotes a spike at time t in run r by cell c.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    :param int window:
        Bin-width for counting spikes, in milliseconds.
    :returns:
        Trial-mean rates of each pattern in each timestep, as a numpy.ndarray
        with `time' rows and sum_{k=1}^{order} {n \choose k} columns, given
        n cells.
    """
    # Get spike-matrix metadata
    T, R, N = spikes.shape
    # Bin spikes
    spikes = spikes.reshape((int(T / window), window, R, N))
    spikes = spikes.any(axis=1)
    # Compute each n-choose-k subset of cell IDs up to `order'
    subsets = enumerate_subsets(N, order)
    # Set up the output array
    y = numpy.zeros((int(T / window), len(subsets)))
    # Iterate over each subset
    for i in range(len(subsets)):
        # Select the cells that are in the subset
        sp = spikes[:,:,subsets[i]]
        # Find the timesteps in which all subset-cells spike coincidentally
        spc = sp.sum(axis=2) == len(subsets[i])
        # Average the occurences of coincident patterns to get the mean rate
        y[:,i] = spc.mean(axis=1)

    return y


def enumerate_subsets(N, O):
    """
    Enumerates all N-choose-k subsets of cell IDs for k = 1, 2, ..., O. For
    example,
        >>> compute_subsets(4, 2)
        >>> [(0,), (1,), (2,), (3,), (0, 1),
                (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    :param int N:
        Total cells from which to choose subsets.
    :param int O:
        Maximum size of subsets to enumerate.
    :returns:
        List of tuples, each tuple containing a subset of cell IDs.
    """
    # Compute each C-choose-k subset of cell IDs up to `O'
    subsets = list()
    ids = numpy.arange(N)
    for k in range(1, O + 1):
        subsets.extend(list(itertools.combinations(ids, k)))
    # Assert that we've got the correct number of subsets
    assert len(subsets) == compute_D(N, O)

    return subsets


def enumerate_patterns(N):
    """
    Enumerates all spike patterns in order, for example:
        >>> enumerate_patterns(3)
        array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 1, 1]], dtype=uint8)
    :param int N:
        Number of cells for which to enumerate patterns.
    :returns:
        Binary matrix of spike patterns as a numpy.ndarray of dimensions
        (2**N, N).
    """
    # Get the spike patterns as ordered subsets
    subsets = enumerate_subsets(N, N)
    assert len(subsets) == 2**N - 1
    # Generate output array and fill according to subsets
    fx = numpy.zeros((2**N, N), dtype=numpy.uint8)
    for i in range(len(subsets)):
        fx[i+1,subsets[i]] = 1

    return fx


def initialise(N, O):
    """
    Sets up matrices to transform between theta, probability and eta
    coordinates. Computing probability requires finding subsets of theta for the
    numerator; for example, with two cells, finding the numerator:
        p(x1,x2) =     e^(t1x1)e^(t2x2)e^(t12x1x2)
                   -----------------------------------
                   1 + e^(t1) + e^(t2) + e^(t1+t2+t12)
    We calculate a `p_map' to do this for arbitrary numbers of neurons and
    orders of interactions. To compute from probabilities to eta coordinates,
    we use an `eta_map' that is just the transpose of the `p_map'. The method
    for doing this is a bit tricky, and not easy to explain. Suffice to say,
    it produces to the correct maps.
    This function has the side effect of setting global variables `p_map' and
    `eta_map', which are used later by other functions in this module.
    :param int N:
        Total cells in the spike data.
    :param int order:
        Order of spike-train interactions to estimate, for example, 2 =
        pairwise, 3 = triplet-wise...
    """
    global p_map, eta_map

    # Create a matrix of binary spike patterns
    fx = enumerate_patterns(N)
    # Compute the number of natural parameters, given the order parameter
    D = compute_D(N, O)
    # Set up the output matrix
    p_map = numpy.ones((2**N, D), dtype=numpy.uint8)
    # Compute the map!
    for i in range(1, D+1):
        idx = numpy.nonzero(fx[i,:])[0]
        for j in range(idx.size):
            p_map[:,i-1] = p_map[:,i-1] & fx[:,idx[j]]
    # Set up the eta map
    eta_map = p_map.transpose()

def compute_eta_LD(eta_tilde, J):
    """
    Computes the expected values of the feature vector of the multi-graph Ising
    model from the supplied expected values of spike patterns and the graphs
    matrix of the model.
    :param numpy.ndarray eta_tilde:
        Expected values of spike patterns for the Ising model with mixed state
        network parameters.
    :param numpy.ndarray J:
        Fitted graphs matrix.
    :returns:
        Expected values of the feature vector of the multi-graph Ising model.
        as a numpy.ndarray.
    """
    # Compute the output array
    eta_LD = numpy.dot(J.T, eta_tilde)
        
    return eta_LD


def compute_fisher_info_LD(fisher_tilde, J):
    """
    Computes the Fisher-informtion matrix of the expected values of the
    feature vector for the multi-graphs Ising model for the purposes of
    Newton-Raphson and computation of the marginal probability distribution
    of the multi-graph Ising model.
    :param numpy.ndarray fisher_tilde:
        Fisher-information matrix for the Ising model with mixed state network
        parameters.
    :param numpy.ndarray J:
        Fitted theta parameters of the original static Ising models.
    :returns:
        Fisher-information matrix for the multi-graph Ising model
        as a numpy.ndarray.
    """
    # Compute the output array
    tmp = numpy.dot(J.T, fisher_tilde)
    fisher_LD = numpy.dot(tmp, J)
        
    return fisher_LD
    

def compute_f_LD(F_tilde, J):
    """
    Computes the feature vector of the multi-graph Ising model for all time
    bins from the empirical mean rate of each spike pattern across trials for
    each timestep.
    :param numpy.ndarray F_tilde:
        Trial-mean rates of each pattern in each timestep.
    :param numpy.ndarray J:
        Fitted graphs matrix.
    :returns:
        Feature vector of the multi-graph Ising model for each timestep as a
        numpy.ndarray.
    """
    f_LD = numpy.dot(F_tilde, J)
            
    return f_LD


def compute_mixed_theta(weight, J):
    """
    Computes the mixed theta parameter from the weights and the graphs matrix.
    :param numpy.ndarray weight:
        Vector of the weights applied to each graph for one time bin (D,).
    :param numpy.ndarray J:
        Fitted graphs matrix.
    :returns:
        Mixed theta parameter (the weighted sum of the original static Ising
        models) as a numpy.ndarray.
    """
    # Compute mixed theta
    theta_mixed = numpy.dot(weight, J.T)
        
    return theta_mixed

