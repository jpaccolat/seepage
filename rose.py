#!/usr/bin/env python

"""
Generate soil samples according to Rosetta3.

For model 1, mean and standard deviations of the log of K, alpha and n are
taken from https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/rosetta-class-average-hydraulic-parameters/
The distribution are assumed gaussian as argued in http://dx.doi.org/10.1016/j.jhydrol.2017.01.004


For model 3 the rosetta-soil package is used.
"""

####################
# Libraries        #
####################

# Standard imports

# Third party imports
import numpy as np
import pandas as pd
from rosetta import Rosetta

####################
# Functions        #
####################

def sample_from_class(name, N, bounds=None):
    """
    Return a dataframe with N values of hydraulic conductivity [m/s], scale [m]
    and shape [-] parameters (from the vGM parametrization), generated according
    to the given textural class (Rosetta3 M1).
    """

    if bounds is None:
        bounds = {
            'K': [0, np.inf],
            'hg': [0, np.inf],
            'n': [1, np.inf]
        }

    p = get_gaussian_parameters(name)

    K = get_hydraulic_conductivity(N, *p['K'], bounds=bounds['K'])
    hg = get_scale_parameter(N, *p['alpha'], bounds=bounds['hg'])
    n = get_shape_parameter(N, *p['n'], bounds=bounds['n'])

    df = pd.DataFrame(np.array([K, hg, n]).T, index=range(N),
                      columns=['K', 'hg', 'n'])

    return df

def sample_from_fraction(N: int, sand: tuple, silt: function, n=20):
    """
    Return a dataframe with N values of hydraulic conductivity [m/s], scale [m]
    and shape [-] parameters (from the vGM parametrization), generated according
    to the sand and silt fraction. Parameters are generated with Rosetta3 M2.
     
    Use int or float (btw. 0 and 100) for a single sand fraction.
    Use a tuple to indicate min and max sand fractions to consider a range of 
    fractions.
    Silt is given as a function of sand fraction.

    Parameters:
    -----------
    N: int
        Sample size
    sand: int or tuple
        Sand fraction, or min and max sand fractions.
    silt: function
        Function returning silt fraction from sand fraction.
    n: int
        Number of sample drawn per sand fraction. Must divide N.
    """

    assert N % n == 0

    rose32 = Rosetta(rosetta_version=3, model_code=2)

    if type(sand) == float or type(sand) == int:
        n = N
        sand = np.array([sand])
    else:
        sand = np.random.uniform(low=sand[0], high=sand[1], size=int(N/n))

    silt = silt(sand)
    clay = 100 - silt - sand
    data = [[*v] for v in zip(sand, silt, clay)]
    retc, ksat = rose32.predict(data)

    idx = np.random.choice(1000, size=n)
    K = 1e-2 / (24 * 3600) * 10**ksat[idx, :, 0].flatten()
    hg = 1e-2 / 10**retc[idx, :, 2].flatten()
    n = 10**retc[idx, :, 3].flatten()

    df = pd.DataFrame(np.array([K, hg, n]).T, index=range(N),
                      columns=['K', 'hg', 'n'])
    
    return df

def get_gaussian_parameters(name):
    """
    Return mean and standard deviation of the Gaussian distriutions of hydraulic
    conductivity, scale (alpha) and shape parameter of the vGM parametrization 
    associated to a textural class according to Rosetta3.
    """

    if name == 'SAND':
        p = {
            'K': [2.808, 0.59],
            'alpha': [-1.453, 0.25],
            'n': [0.502, 0.18]
        }
    
    if name == 'LOAM_SAND':

        p = {
            'K': [2.022, 0.64],
            'alpha': [-1.459, 0.47],
            'n': [0.242, 0.16]
        }

    if name == 'SAND_LOAM':

        p = {
            'K': [1.583, 0.66],
            'alpha': [-1.574, 0.56],
            'n': [0.161, 0.11]
        }

    if name == 'LOAM':

        p = {
            'K': [1.081, 0.92],
            'alpha': [-1.954, 0.73],
            'n': [0.168, 0.13]
        }

    if name == 'SILT_LOAM':
        p = {
            'K': [1.261, 0.74],
            'alpha': [-2.296, 0.57],
            'n': [0.221, 0.14]
        }

    if name == 'SILT':
        p = {
            'K': [1.641, 0.27],
            'alpha': [-2.182, 0.30],
            'n': [0.225, 0.13]
        }

    if name == 'CLAY':
        p = {
            'K': [1.641, 0.27],
            'alpha': [-2.182, 0.30],
            'n': [0.225, 0.13]
        }


    return p

def get_hydraulic_conductivity(N, mean, std, bounds=[0, np.inf]):
    """
    Return a list of normaly distributed hydraulic conductivities in m/s. Bounds
    on the distribution can be imposed.
    """

    larger_N = 2 * N
    while True:
        log_K = np.random.normal(loc=mean, scale=std, size=larger_N)
        K = 1e-2 / (24 * 60 * 60) * 10**log_K

        idx = np.logical_and(K > bounds[0], K < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            K = K[idx]
            break
    
    return K[:N]

def get_scale_parameter(N, mean, std, bounds=[0, np.inf]):
    """
    Return a list of normaly distributed scale parameters (hg) in m. Bounds on
    the distribution can be imposed.
    """

    larger_N = 2 * N
    while True:
        log_alpha = np.random.normal(loc=mean, scale=std, size=larger_N)
        hg = 1e-2 / 10**log_alpha

        idx = np.logical_and(hg > bounds[0], hg < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            hg = hg[idx]
            break
    
    return hg[:N]

def get_shape_parameter(N, mean, std, bounds=[1, np.inf]):
    """
    Return a list of normaly distributed shape parameters (n). Bounds on the
    distribution can be imposed.
    """

    larger_N = 2 * N
    while True:
        log_n = np.random.normal(loc=mean, scale=std, size=larger_N)
        n = 10**log_n

        idx = np.logical_and(n > bounds[0], n < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            n = n[idx]
            break
    
    return n[:N]