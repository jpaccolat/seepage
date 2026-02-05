#!/usr/bin/env python

"""
Generate soil samples according to Rosetta3 H1 model.

Mean and standard deviations of the log of K, alpha and n are taken from
https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/rosetta-class-average-hydraulic-parameters/

The distribution are assumed gaussian as argued in
http://dx.doi.org/10.1016/j.jhydrol.2017.01.004
"""

####################
# Libraries        #
####################

# Standard imports

# Third party imports
import numpy as np
import pandas as pd

####################
# Constants        #
####################

SAND_K_mean = 2.808
SAND_K_std = 0.59
SAND_n_mean = 0.502
SAND_n_std = 0.18
SAND_alpha_mean = -1.453
SAND_alpha_std = 0.25

SILT_K_mean = 1.641
SILT_K_std = 0.27
SILT_n_mean = 0.225
SILT_n_std = 0.13
SILT_alpha_mean = -2.182
SILT_alpha_std = 0.30

####################
# Functions        #
####################

def get_sand(N):
    """Generate N samples of sandy soil."""

    log_K = np.random.normal(loc=SAND_K_mean, scale=SAND_K_std, size=N)
    log_n = np.random.normal(loc=SAND_n_mean, scale=SAND_n_std, size=N)
    log_alpha = np.random.normal(loc=SAND_alpha_mean, scale=SAND_alpha_std,
                                 size=N)
    
    df = pd.DataFrame(columns=['K', 'hg', 'n'])
    df['K'] = 1e-2 / (24 * 60 * 60) * 10**log_K
    df['n'] = 10**log_n
    df['hg'] = 1e-2 / 10**log_alpha

    return df

def get_silt(N):
    """Generate N samples of silty soil."""

    log_K = np.random.normal(loc=SILT_K_mean, scale=SILT_K_std, size=N)
    log_n = np.random.normal(loc=SILT_n_mean, scale=SILT_n_std, size=N)
    log_alpha = np.random.normal(loc=SILT_alpha_mean, scale=SILT_alpha_std,
                                 size=N)
    
    df = pd.DataFrame(columns=['K', 'hg', 'n'])
    df['K'] = 1e-2 / (24 * 60 * 60) * 10**log_K
    df['n'] = 10**log_n
    df['hg'] = 1e-2 / 10**log_alpha

    return df 