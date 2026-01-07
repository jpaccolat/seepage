#!/usr/bin/env python

"""Unsaturated hydraulic conductivity (UHC) module.

This module contains the unsaturated hydraulic conductivity functions for 
 - vgm (Van Genuchten - Mualem)
 - bcb (Brooks and Corey - Burdine)
parametrizations.

The associated water retention functions are also defined.
"""

####################
# Libraries        #
####################

# Standard imports

# Third party imports
import numpy as np

# Internal imports

####################
# Functions        #
####################

def get_rhc(para: str):
    """
    Return relative hydraulic conductivity function corresponding to the given
    parametrization key.
    
    Parameters
    ----------
    para: str
        Parametrization key. Either vGM (for van Genuchten - Mualem) or BCB
        (for Brooks and Corey - Burdine).
    """

    if para == 'vGM':
        return relative_hydraulic_conductivity_vGM
    elif para == 'BCB':
        return relative_hydraulic_conductivity_BCB
    else:
        print("Unkown parametrization. Use either vGM (van Genuchten - Mualem) " 
              "or BCB (Brooks and Corey - Burdine).") 


def water_retention_vG(psi, hg, n, m):
    """
    Compute the water retention according to van Genucthen parametrization

    Parameters
    ----------
    psi: np.array
        Suction [L] (positive).
    hg: float
        Scale parameter [L] (positive).
    n:  float.
        Frist shape parameter [-].
    m:  float
        Second shape parameter [-].

    Returns
    -------
    np.array
        Degree of saturation.
    """

    S = np.ones_like(psi)
    S[psi > 0] = (1 + (psi[psi > 0] / hg)**n)**(-m)

    return S

def water_retention_BC(psi, hg, lam):
    """
    Compute the water retention according to Brooks and Corey parametrization

    Parameters
    ----------
    psi: np.array
        Suction [L] (positive).
    hg: float
        Scale parameter [L] (positive).
    lam:  float.
        Shape parameter [-].

    Returns
    -------
    np.array
        Degree of saturation.
    """

    S = np.ones_like(psi)
    S[psi > hg] = (psi[psi > hg] / hg)**(-lam)

    return S

def relative_hydraulic_conductivity_vGM(psi, hg, n):
    """
    Compute the relative unsaturated hydraulic conductivty according to
    van Genuchten - Mualem theory.

    Parameters
    ----------
    psi: np.array
        Suction [L] (positive).
    hg: float
        Scale parameter [L] (positive).
    n:  float.
        Shape parameter [-] (n > 1).

    Returns
    -------
    np.array
        Relative hydraulic conductivity.
    """

    m = 1 - 1 / n
    S = water_retention_vG(psi, hg, n, m)

    return S**0.5 * (1 - (1 - S**(1 / m))**m)**2

def relative_hydraulic_conductivity_BCB(psi, hg, lam):
    """
    Compute the relative unsaturated hydraulic conductivty according to
    Brooks and Corey - Burdine theory.

    Parameters
    ----------
    psi: np.array
        Suction [L] (positive).
    hg: float
        Scale parameter [L] (positive).
    lam:  float.
        Shape parameter [-] (positive).

    Returns
    -------
    np.array
        Relative hydraulic conductivity.
    """

    rhc = np.ones_like(psi)
    exp = -(2 + 3 * lam)
    rhc[psi > hg] = (psi[psi > hg] / hg)**exp

    return rhc