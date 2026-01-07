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
from time import time

# Third party imports
import numpy as np
from scipy.optimize import fsolve

# Internal imports
import constants
from uhc import get_rhc

####################
# Functions        #
####################

def q_exact(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    rhc = get_rhc(aq_para)
    
    def darcy(psi_interface):
        lhs = aq_cond * rhc(psi_interface, aq_scale, aq_shape)
        rhs = cl_cond * (1 + (stage + psi_interface) / cl_th)
        return lhs - rhs

    psi_interface_init = aq_scale
    x, _, ier, _ = fsolve(darcy, psi_interface_init, full_output=True)
    psi_interface = x[0] if ier == 1 else np.nan
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

def q0_asymptote_negl(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = aq_cond

    return q0

def q0_asymptote_soft(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    b, B = get_bB()
    x = get_x()
    q0 = aq_cond * x**-(b/(1+b))
    
    return q0

def q0_asymptote_hard(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    q0 = cl_cond
    
    return q0

def q0_approximate(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    
    x, x_sh, b, B, xi = get_dless_parameters()

    if aq_para == 'vGM':
        
        a_ns = np.log(constants.C1 * x)
        q_ns = aq_cond * (1 + x**a_ns)**-(xi / a_ns)
        
        a_sh = 0.5 * (constants.C2 + aq_shape)
        q_sh = cl_cond * (1 + (x_sh / x)**a_sh)**(xi / a_sh)
        
        s = 1 / (1 + (x_sh**0.5 / x)**constants.A)
        q0 = q_ns**(1-s) * q_sh**s

    elif aq_para == 'BCB':

        a_sh = 0.5 * (constants.C2 + aq_shape)
        q_sh = cl_cond * (1 + (x_sh / x)**a_sh)**(xi / a_sh)

        q0 = max(aq_cond, q_sh)

    return q0

def q_approximate(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):

    q0 = q0_approximate(cl_cond=cl_cond, cl_th=cl_th, aq_cond=aq_cond,
            aq_scale=aq_scale, aq_shape=aq_shape, aq_para=aq_para)
    q = q0 + stage / cl_th * cl_cond

    return q

def q_modflow(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):

    q = cl_cond * (1 + stage / cl_th)

    return q