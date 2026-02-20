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

# from soiltexture import getTextures
# from rosetta import Rosetta

####################
# Functions        #
####################

def get_gaussian_parameters(name):

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


    return p

def get_hydraulic_conductivity(N, mean, std, bounds=[0, np.inf]):

    larger_N = 2 * N
    while True:
        log_K = np.random.normal(loc=mean, scale=std, size=larger_N)
        K = 1e-2 / (24 * 60 * 60) * 10**log_K

        idx = np.logical_and(K > bounds[0], K < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            break
    
    return K[:N]

def get_scale_parameter(N, mean, std, bounds=[0, np.inf]):

    larger_N = 2 * N
    while True:
        log_alpha = np.random.normal(loc=mean, scale=std, size=larger_N)
        hg = 1e-2 / 10**log_alpha

        idx = np.logical_and(hg > bounds[0], hg < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            break
    
    return hg[:N]

def get_shape_parameter(N, mean, std, bounds=[1, np.inf]):

    larger_N = 2 * N
    while True:
        log_n = np.random.normal(loc=mean, scale=std, size=larger_N)
        n = 10**log_n

        idx = np.logical_and(n > bounds[0], n < bounds[1])
        if idx.sum() < N: 
            larger_N *= 2
        else:
            break
    
    return n[:N]

def sample_soils(name, N, bounds=None):

    if bounds is None:
        bounds = {
            'K': [0, np.inf],
            'hg': [0, np.inf],
            'n': [1, np.inf]
        }

    p = get_gaussian_parameters(name)

    K = get_hydraulic_conductivity(N, *p['K'], bounds['K'])
    hg = get_scale_parameter(N, *p['alpha'], bounds['hg'])
    n = get_shape_parameter(N, *p['n'], bounds['n'])

    df = pd.DataFrame(np.array([K, hg, n]).T, index=range(N),
                      columns=['K', 'hg', 'n'])

    return df

# model 2 of Rosetta3

# def sample_texture(N, texture_names):

#     fractions = sample_texture_triangle(N, texture_names)
#     df = sample_unsat_parameters(fractions)

#     return df

# def sample_texture_triangle(N, texture_names):
#     """
#     0=none
#     1=clay
#     2=silty clay
#     3=silty clay loam
#     4=sandy clay 
#     5=sandy clay loam
#     6=clay loam
#     7=silt
#     8=silt loam
#     9=loam
#     10=sand 
#     11=loamy sand
#     12=sandy loam
#     13=silt loam
#     """

#     fractions = np.zeros((1, 3))

#     while True:
#         sand_frac = np.random.uniform(low=0, high=100, size=N)
#         clay_frac = np.random.uniform(low=0, high=100, size=N)

#         textures = getTextures(sand_frac, clay_frac, classification='USDA')
    
#         idx = np.zeros(N, dtype=bool)
#         for name in texture_names:
#             idx = np.logical_or(idx, np.array(textures) == name)

#         if idx.sum() == 0: continue

#         new_fractions = np.array([
#                 sand_frac[idx],
#                 100 - sand_frac[idx] - clay_frac[idx],
#                 clay_frac[idx]
#             ]).T
#         fractions = np.append(fractions, new_fractions, axis=0)
        
#         if len(fractions) > N: break

#     fractions = np.delete(fractions, 0, axis=0)

#     return fractions

# def sample_unsat_parameters(fractions):

#     df = pd.DataFrame(index=range(len(fractions)), columns=['K', 'hg', 'n'])

#     rose32 = Rosetta(rosetta_version=3, model_code=2)
#     x = rose32.ann_predict(fractions, sum_data=False)['res'][0]

#     idx = np.random.randint(0, high=1000, size=len(fractions))
#     for i, idx_i in enumerate(idx):
#         K = 1e-2 / (24 * 60 * 60) * 10**x[idx_i, 4, i].flatten()
#         hg = 1e-2 / 10**x[idx_i, 2, i].flatten()
#         n = 10**x[idx_i, 3, i].flatten()
#         df.loc[i] = [K, hg, n]

#     return df