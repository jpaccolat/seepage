#!/usr/bin/env python

"""

"""

####################
# Libraries        #
####################

# Standard imports
import sys
from math import isclose
from time import time

# Third party imports
import numpy as np
from scipy import stats
from alphashape import alphashape
from shapely import geometry

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch

# Internal imports
sys.path.append('..')
from rose import sample_from_class, sample_from_fraction
from dless import get_dless_parameters

####################
# Constants        #
####################

g = 9.81 # [m/s^2]
density = 999.1 # [kg/m^3] at 15 °C
viscosity = 1.14e-6 # [m^2/s] at 15 °C
surface_tension = 0.0735 # [N/m] at 15 °C

####################
# Functions        #
####################

def get_region(sb, N=1000, n=20):
    """
    Return Artist that represent a one- or two-dimensional region in the 
    infiltrability space.

    
    First, generate arrays of size N for clogging and aquifer parameters.
    Second, convert to infiltrability coordinates (x, x_sh).
    Third, compute dimensionality. If D=1, return a Line2D; if D=2, return 
    a Polygon (concave hull of the points).


    If cl_texture (or aq_texture) is None:
    (1) cl_cond (aq_cond) is single valued -> identical array
    (2) cl_cond (aq_cond) is double valued -> draw from loguniform
    In both cases, if aq_scale and aq_shape are None, they are computed from 
    aq_cond according to Peche2024 (10.1111/gwat.13365).

    If cl_texture (or aq_texture) is a string, parameters are sampled from the
    given textural class with Rosetta3 M1. Only the conductivity is sampled for
    the clogging layer, but also the scale and shape parameters for the aquifer.

    If cl_texture (or aq_texture) is a tuple, the values correspond to the
    bounds on the sand fraction. Parameters are sampled by varying it (with 
    zero clay content), with N / n steps. At each step, n samples are drawn 
    with Rosetta3 M2. Only the conductivity is sampled for the clogging layer,
    but also the scale and shape parameters for the aquifer.
    """

    assert sb['type'] == 'region'

    # generate clogging layer
    if len(sb['cl_th']) == 1:
        cl_th = sb['cl_th'][0] * np.ones(N)
    else:
        min, max = sb['cl_th']
        cl_th = np.random.uniform(low=min, high=max, size=N)

    if sb['cl_texture'] is None:
        if len(sb['cl_cond']) == 1:
            cl_cond = sb['cl_cond'][0] * np.ones(N)
        else:
            min, max = sb['cl_cond']
            cl_cond = 10**np.random.uniform(low=np.log10(min),
                                            high=np.log10(max),
                                            size=N)
    elif type(sb['cl_texture']) == str:
        df = sample_from_class(sb['cl_texture'], N)
        cl_cond = df['K'].values
    else:
        f_silt = lambda x: 100 - x
        df = sample_from_fraction(sb['cl_texture'], f_silt, N, n)
        cl_cond = df['K'].values


    # generate aquifer
    if sb['aq_texture'] is None:
        if len(sb['aq_cond']) == 1:
            aq_cond = sb['aq_cond'][0] * np.ones(N)
        else:
            min, max = sb['aq_cond']
            aq_cond = 10**np.random.uniform(low=np.log10(min),
                                            high=np.log10(max),
                                            size=N)
    
        if sb['aq_scale'] is None:
            aq_scale = 1 / get_alpha_v(aq_cond)
        elif len(sb['aq_scale']) == 1:
            aq_scale = sb['aq_scale'][0] * np.ones(N)
        else: 
            min, max = sb['aq_scale']
            aq_scale = np.random.uniform(low=min, high=max, size=N)

        if sb['aq_shape'] is None:
            aq_shape = get_n_v(1 / aq_scale)
        elif len(sb['aq_shape']) == 1:
            aq_shape = sb['aq_shape'][0] * np.ones(N)
        else: 
            min, max = sb['aq_shape']
            aq_shape = np.random.uniform(low=min, high=max, size=N)

    elif type(sb['aq_texture']) == str:
        df = sample_from_class(sb['aq_texture'], N)
        aq_cond = df['K'].values
        aq_scale = df['hg'].values
        aq_shape = df['n'].values
    else:
        f_silt = lambda x: 100 - x
        df = sample_from_fraction(sb['aq_texture'], f_silt, N, n)
        aq_cond = df['K'].values
        aq_scale = df['hg'].values
        aq_shape = df['n'].values
    

    # convert to infiltrability space
    _, _, _, x, x_sh = get_dless_parameters(cl_cond, cl_th, aq_cond, aq_scale,
                                            aq_shape, sb['aq_para'])
    points = np.array([np.log10(x), np.log10(x_sh)])

    # build Artist (Line2D id D=1, Polygon if D=2)
    dimension = get_dimension(points)
    if dimension == 1:
        x, y = points
        if is_on_line(points):
            imin, imax = np.argmin(x), np.argmax(x)    
            shape = Line2D([x[imin], x[imax]], [y[imin], y[imax]])
        else:
            idx = np.argsort(x)
            shape = Line2D(x[idx], y[idx])
    else:
        alpha = 2
        while True:
            a = alphashape(points.T, alpha)
            if type(a) == geometry.polygon.Polygon:
                vertices = a.exterior.xy
                shape = Polygon(np.array([vertices[0], vertices[1]]).T)
                break
            else:
                alpha -= 0.5
            # if alpha is too big, construct MultiPolygon. Reduce it to get a
            # Polygon.
                
    return dimension, shape

def get_points(sb):
    """
    Return infiltrability coordinates of the streambeds.

    cl_cond, cl_th and aq_cond must have the same length. 
    If aq_scale and/or aq_shape are None, they are computed from aq_cond
    according to Peche2024 (10.1111/gwat.13365).
    """

    assert sb['type'] == "points"

    cl_cond = sb['cl_cond']
    cl_th = sb['cl_th']
    aq_cond = sb['aq_cond']

    if sb['aq_scale'] is None:
        aq_scale = 1 / get_alpha_v(aq_cond)
    else:
        aq_scale = sb['aq_scale']

    if sb['aq_shape'] is None:
        aq_shape = get_n_v(aq_scale)
    else:
        aq_shape = sb['aq_shape']
        
    points = []
    for v in zip(cl_cond, cl_th, aq_cond, aq_scale, aq_shape):
        _, _, _, x, x_sh = get_dless_parameters(*v, 'vGM')
        points.append([np.log10(x), np.log10(x_sh)])
    
    return np.array(points).T

def get_dimension(points):
    """
    Compute the dimensionality of a set of points in 2D. If the Spearman
    correlation is one, the dimensionality is one. Else it is two.
    """

    x, y = points
    if all(x == x[0]) or all(y == y[0]):
        return 1
    r = stats.spearmanr(points.T).statistic
    if r == 1:
        return 1
    else:
        return 2

def is_on_line(points):
    """Check if a set of points fall on the same line."""
    x, y = points
    idx = x != x[0]
    slopes = (y[idx] - y[0]) / (x[idx] - x[0])
    c1 = all(isclose(s, slopes[0], rel_tol=1e-3) for s in slopes)
    c2 = all(y[~idx] == y[0])
    return 2 - c1 * c2

def plot_streambed(x, ax, labels, color):
    """Plot the points or Artist associated to the given study."""

    print(x['name'])
    if x['source'] == 'synthetic':
        r = '$^\\star$'
    elif x['source'] == 'river':
        r = '$^r$'
    else:
        r = '$^s$'
    
    if x['type'] == "points":
        points = get_points(x)      
        ax.scatter(*points, s=20, marker='o', fc=color, ec='k', zorder=3)

    if x['type'] == "region":
        dimension, region = get_region(x)
        if dimension == 1:
            region.set_color(color)
            region.set_linewidth(2)
            region.set_zorder(2)
            ax.add_line(region)
        else:
            region.set_edgecolor('k')
            region.set_facecolor(color)
            region.set_alpha(0.8)
            region.set_zorder(1)
            ax.add_patch(region)

    if x['n'] == 1:
        labels.append(Patch(fc=color, ec='k', label=f'{r}{x['name']}'))

def get_d60(K):
    """
    Get d60 fraction from hydraulic conductivity.
    Expression from Peche2024 (10.1111/gwat.13365)
    """

    Pi = 0.0009
    c1 = 1.2
    c2 = 1.13

    d10 = np.sqrt(K / Pi * viscosity / g)
    d20 = c1 * d10
    ne_new = 1.004 * K**(1.51e-4)
    d50_new = 5.788e-3 * d20

    t1 = time()
    while True:
        ne_old = ne_new
        d50_old = d50_new
        ne_new = (K / d50_old**2 * 180 * viscosity / g * (1 - ne_old)**2)**(1/3)
        d50_new = (K * 180 * viscosity / g * (1 - ne_new)**2 / ne_new**3)**.5
        if time() - t1 > 1:
            break
        if (ne_new**2-ne_old**2)**.5 + (d50_new**2-d50_old**2)**.5 < 1e-4:
            break

    d60 = c2 * d50_new

    return d60

def get_alpha(K):
    """
    Get van Genuchten alpha from hydraulic conductivity.
    Expression from Peche2024 (10.1111/gwat.13365)
    """

    c1 = 1.803e-4
    c2 = 0.1

    dp = get_d60(K) if K < 5e-5 else c1 + c2 * K
    hg = 2 * surface_tension / (density * g) / (dp / 2)
    return 1 / hg
get_alpha_v = np.vectorize(get_alpha)

def get_n(alpha):
    """
    Get van Genuchten n from alpha.
    Expression from Peche2024 (10.1111/gwat.13365)
    """

    c3 = 1.124
    c4 = 0.557
    c5 = 1.676
    c6 = 0.303
    c7 = 1.162

    return c3 + c4 * alpha if alpha < 1.9 else c5 / (alpha - c6) + c7
get_n_v = np.vectorize(get_n)