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
from scipy.integrate import solve_bvp

# Internal imports
from uhc import get_rhc

#############
# Functions #
#############

def q_exact(depth: float, stage: float, cl_cond: float, cl_th: float,
            aq_cond: float, aq_scale: float, aq_shape: float, aq_para: str,
            guess=0., max_nodes=1000, tol=1e-3):
    
    """
    Solve the steady state Richard's equation for given boundary conditions.
    The solution is stored as a class attribute.

    Parameters
    ----------
    guess: float, optional
        Initial guess for the infiltration rate in [m/s]. Default is 0.
    max_nodes: int, optional
        Maximal number of nodes. Default is 1000.
    tol: float, optional
        Tolerance of the BVP solver. Default is 1e-3.
    """

    rhc = get_rhc(aq_para)

    def fun(x, y, p):

        psi = -y[0]
        cond = cl_cond * np.ones_like(psi)

        idx = x > cl_th
        cond[idx] = aq_cond * rhc(psi[idx], aq_scale, aq_shape)

        return np.vstack([1 - p[0] / cond])

    def bc(ya, yb, p):
        return np.array([ya[0] - stage, yb[0]])

    z = np.linspace(0, cl_th + depth, 10)
    y = np.zeros((1, z.size))
    y[0, 0] = stage
    y[0, -1] = 0.
    
    sol = solve_bvp(fun, bc, z, y, p=[guess], max_nodes=max_nodes, tol=tol)

    return sol.p[0]

def q_exact_full(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
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

def q_approx(depth: float, stage: float, cl_cond: float, cl_th: float,
             aq_cond: float, aq_scale: float, aq_shape: float, aq_para: str):
    
    q1 = (stage + cl_th + depth) / (cl_th / cl_cond + depth / aq_cond)
    q2 = q_approx_full(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                       aq_para)
    q = np.min([q1, q2], axis=0)

    return q

def q_approx_full(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
                  aq_scale: float, aq_shape: float, aq_para: str):
    
    if aq_para == 'vGM':
        q = q_approx_full_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape)
    elif aq_para == 'BCB':
        q = q_approx_full_BCB(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape)
        
    return q

def q_approx_full_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape):
    
    q0 = q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    b = 0.5 * (5 * aq_shape - 1)
    hc = cl_th * (aq_cond / cl_cond - 1)
    x = np.min([0.99 * aq_cond, (1 + b) * q0], axis=0)
    s = -cl_cond / cl_th * (q0 - cl_cond) / (x - cl_cond)
    hstar =  (q0 - cl_cond) / (s * hc + q0 - cl_cond) * hc
    q = q0 + (cl_cond / cl_th - s * hstar / (stage - hstar)) * stage

    return q

def q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape, 
                       C_NS_1=-0.3850, C_NS_2=0.2056, C_NS_3=0.5818,
                       C_SH_1=0.4633, C_SH_2=0.5396, C_NSH_1=1.05, C_NSH_2=0.25,
                       C_NSH_3=6):

    b = 0.5 * (5 * aq_shape - 1)
    B = (1 - 1 / aq_shape)**2

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_ns = (C_NS_1 + C_NS_2 * b)**C_NS_3
    q0_ns = q0_negl_to_soft(aq_cond, x, xi, a_ns)
    
    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    a_nsh = C_NSH_1 + C_NSH_2 * np.tanh(b - C_NSH_3)
    s = 1 / (1 + (x_sh**0.5 / x)**a_nsh)
    q0 = q0_ns**(1-s) * q0_sh**s

    return q0

def q_approx_full_BCB(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape):

    q0 = q0_approx_full_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    b = 2 + 3 * aq_shape
    s = -cl_cond / cl_th * (q0 - cl_cond) / ((1+b) * q0 - cl_cond)
    hc = cl_th * (aq_cond / cl_cond - 1) - aq_scale
    cl_cond_cor = cl_cond * (1 + aq_scale / cl_th) 
    hstar =  (q0 - cl_cond_cor) / (s * hc + q0 - cl_cond_cor) * hc
    q = q0 + (cl_cond / cl_th - s * hstar / (stage - hstar)) * stage

    return q

def q0_approx_full_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                       C_SH_1=0.4633, C_SH_2=0.5396):

    b = 2 + 3 * aq_shape
    B = 1

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    q0 = np.min([aq_cond, q0_sh], axis=0)

    return q0

def q0_negl_to_soft(aq_cond, x, xi, a_ns):
    return aq_cond * (1 + x**a_ns)**(-xi/a_ns)

def q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh):
    return cl_cond * (1 + (x_sh / x)**a_sh)**(xi / a_sh)

def q_modflow(stage: float, cl_cond: float, cl_th: float):

    q = cl_cond * (1 + stage / cl_th)

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