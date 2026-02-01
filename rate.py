#!/usr/bin/env python

"""Rate module.

This module defines functions to compute the following seepage rates:
 - q0_exact
 - q0_approximate
 - q0_negligible
 - q0_soft
 - q0_hard
 - q_exact
 - q_linear (q0_exact + stage / resistance)
 - q_approximate (q0_approximate + stage / resistance)
 - q_modflow
"""

####################
# Libraries        #
####################

# Standard imports
import time

# Third party imports
import numpy as np
from scipy.optimize import fsolve

# Internal imports
from uhc import get_rhc

####################
# Functions        #
####################

def q_exact(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    """
    Compute exact disconnected seepage rate by solving the unsaturated Darcy 
    equation.

    Parameters:
    stage (float)
        water depth in the river [L].
    cl_cond (float):
        Hydraulic conductivity of the clogging layer [L/T].
    cl_th (float):
        Thickness of the clogging layer [L].
    aq_cond (float)
        Hydraulic conductivity of the aquifer [L/T].
    aq_scale (float)
        Scale parameter of the aquifer [L].
    aq_shape (float)
        Shape parameter of the aquifer [-].
    aq_para (str)
        Unsaturated hydraulic conductivity parametrization.
    """
    
    rhc = get_rhc(aq_para)
    
    def darcy(psi_interface):
        lhs = aq_cond * rhc(psi_interface, aq_scale, aq_shape)
        rhs = cl_cond * (1 + (stage + psi_interface) / cl_th)
        return lhs - rhs
        
    psi_interface_init = aq_scale
    x, _, ier, _ = fsolve(darcy, psi_interface_init, full_output=True)
    #psi_interface = x[0] if ier == 1 else np.nan
    psi_interface = x[0]
    q = cl_cond * (1 + (stage + psi_interface) / cl_th)

    return q

def q0_exact(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = q_exact(stage=0., cl_cond=cl_cond, cl_th=cl_th, aq_cond=aq_cond,
            aq_scale=aq_scale, aq_shape=aq_shape, aq_para=aq_para)
        
    return q0

def q_linear(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = q0_exact(cl_cond=cl_cond, cl_th=cl_th, aq_cond=aq_cond,
            aq_scale=aq_scale, aq_shape=aq_shape, aq_para=aq_para)
    q = q0 + stage / cl_th * cl_cond
    
    return q

def q0_asymptote_negl(cl_cond: float,  cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = aq_cond

    return q0

def q0_asymptote_soft(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    if aq_para == 'vGM':
        b = 0.5 * (5 * aq_shape - 1)
        B = (1 - 1 / aq_shape)**2
    elif aq_para == 'BCB':
        b = 2 + 3 * aq_shape
        B = 1

    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    q0 = aq_cond * x**-(b/(1+b))
    
    return q0

def q0_asymptote_hard(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = cl_cond
    
    return q0

def q0_negl_to_soft(aq_cond, x, xi, a_ns):
    return aq_cond * (1 + x**a_ns)**(-xi/a_ns)

def q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh):
    return cl_cond * (1 + (x_sh / x)**a_sh)**(xi / a_sh)

def q0_approximate_vGM(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, C_NSH: float,
             C_NS_1=-0.3850, C_NS_2=0.2056, C_NS_3=0.5818, C_SH_1=0.4633,
             C_SH_2=0.5396):

    b = 0.5 * (5 * aq_shape - 1)
    B = (1 - 1 / aq_shape)**2

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_ns = (C_NS_1 + C_NS_2 * b)**C_NS_3
    q0_ns = q0_negl_to_soft(aq_cond, x, xi, a_ns)
    
    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    s = 1 / (1 + (x_sh**0.5 / x)**C_NSH)
    q0 = q0_ns**(1-s) * q0_sh**s

    return q0

def q0_approximate_BCB(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, C_SH_1=0.4633, C_SH_2=0.5396):

    b = 2 + 3 * aq_shape
    B = 1

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    q0 = min(aq_cond, q0_sh)

    return q0

def q0_approximate(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str, C_NSH=None):
    
    if aq_para == 'vGM':
        q0 = q0_approximate_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                                C_NSH=C_NSH)
    elif aq_para == 'BCB':
        q0 = q0_approximate_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    return q0

def q_approximate(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str, C_NSH=None):
    
    if aq_para == 'vGM':
        q = q_approximate_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape, C_NSH=C_NSH)
    elif aq_para == 'BCB':
        q = q_approximate_BCB(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape)

    return q

def q_approximate_vGM(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, C_NSH=1.):
    
    q0 = q0_approximate_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                            C_NSH=C_NSH)
    
    b = 0.5 * (5 * aq_shape - 1)
    s = q0 / cl_th * b / ((1 + b) * q0 / cl_cond - 1)
    q = max(q0 + s * stage, cl_cond + cl_cond / cl_th * stage)

    # hc = cl_th * (aq_cond / cl_cond - 1)
    # A2 = cl_cond
    # B2 = cl_cond / cl_th
    # A1 = q0 - A2
    # B1 = -A1 / hc
    # hstar = A1 / (B1 + B2 - s)
    # q = (A1 + B1 * stage) * np.exp(-stage/hstar) + (A2 + B2 * stage)

    return q

def q_approximate_BCB(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float):

    q0 = q0_approximate_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    b = 2 + 3 * aq_shape
    s = q0 / cl_th * b / ((1 + b) * q0 / cl_cond - 1)
    B = cl_cond / cl_th
    A = s - B
    hc = cl_th * (aq_cond / cl_cond - 1) - aq_scale
    hstar = -hc / np.log(((aq_cond - q0) / hc - B) / A) 
    q = q0 + (A * np.exp(-stage / hstar) + B) * stage
    
    return q

def q_modflow(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):

    q = cl_cond * (1 + stage / cl_th)

    return q


