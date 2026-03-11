#!/usr/bin/env python

"""Seepage rate formulas.

All exact and approximate formulas to compute seepage are listed here.
"""

####################
# Libraries        #
####################

# Standard imports

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
    Exact seepage through a clogging layer with a shallow water table.

    Solve the steady state Richard's equation for given boundary conditions.

    Parameters
    ----------
    depth: float
        Depth to the water table [m].
    stage: float
        Water depth in the stream [m].
    cl_cond: float
        Hydraulic conductivity of the clogging layer [m/s].
    cl_th: float
        Thickness of the clogging layer [m].
    aq_cond: float
        Hydraulic conductivity of the underlying aquifer [m/s].
    aq_scale: float
        Scale parameter of the unsaturated parametrization of the underlying
        aquifer [m].
    aq_shape: float
        Shape parameter of the unsaturated parametrization of the underlying
        aquifer [-].
    aq_para: str
        Name of the unsaturated parametrization of the underlying aquifer.
        Either vGM (van Genuchten - Mualem) or BCB (Brooks - Corey - Burdine).
    guess: float, optional
        Initial guess of seepage rate. Default is 0.
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
q_exact = np.vectorize(q_exact)

def q_exact_full(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    """
    Exact seepage through a clogging layer at full disconnection.

    Solve the unsaturated Darcy equation.

    Parameters
    ----------
    stage: float
        Water depth in the stream [m].
    cl_cond: float
        Hydraulic conductivity of the clogging layer [m/s].
    cl_th: float
        Thickness of the clogging layer [m].
    aq_cond: float
        Hydraulic conductivity of the underlying aquifer [m/s].
    aq_scale: float
        Scale parameter of the unsaturated parametrization of the underlying
        aquifer [m].
    aq_shape: float
        Shape parameter of the unsaturated parametrization of the underlying
        aquifer [-].
    aq_para: str
        Name of the unsaturated parametrization of the underlying aquifer.
        Either vGM (van Genuchten - Mualem) or BCB (Brooks - Corey - Burdine).
    """
    
    rhc = get_rhc(aq_para)
    
    def darcy(psi_interface):
        lhs = aq_cond * rhc(psi_interface, aq_scale, aq_shape)
        rhs = cl_cond * (1 + (stage + psi_interface) / cl_th)
        return lhs - rhs
        
    psi_interface_init = aq_scale
    x, _, _, _ = fsolve(darcy, psi_interface_init, full_output=True)
    psi_interface = x[0]
    q = cl_cond * (1 + (stage + psi_interface) / cl_th)

    return q
q_exact_full = np.vectorize(q_exact_full)

def q_approx(depth: float, stage: float, cl_cond: float, cl_th: float,
             aq_cond: float, aq_scale: float, aq_shape: float, aq_para: str):
    """
    Approximate seepage through a clogging layer with a shallow water table.

    Parameters
    ----------
    depth: float
        Depth to the water table [m].
    stage: float
        Water depth in the stream [m].
    cl_cond: float
        Hydraulic conductivity of the clogging layer [m/s].
    cl_th: float
        Thickness of the clogging layer [m].
    aq_cond: float
        Hydraulic conductivity of the underlying aquifer [m/s].
    aq_scale: float
        Scale parameter of the unsaturated parametrization of the underlying
        aquifer [m].
    aq_shape: float
        Shape parameter of the unsaturated parametrization of the underlying
        aquifer [-].
    aq_para: str
        Name of the unsaturated parametrization of the underlying aquifer.
        Either vGM (van Genuchten - Mualem) or BCB (Brooks - Corey - Burdine).
    """
    
    q1 = (stage + cl_th + depth) / (cl_th / cl_cond + depth / aq_cond)
    q2 = q_approx_full(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                       aq_para)
    q = np.min([q1, q2], axis=0)

    return q

def q_approx_full(stage: float, cl_cond: float, cl_th: float, aq_cond: float,
                  aq_scale: float, aq_shape: float, aq_para: str):
    """
    Approximate seepage through a clogging layer at full disconnection.

    Parameters
    ----------
    stage: float
        Water depth in the stream [m].
    cl_cond: float
        Hydraulic conductivity of the clogging layer [m/s].
    cl_th: float
        Thickness of the clogging layer [m].
    aq_cond: float
        Hydraulic conductivity of the underlying aquifer [m/s].
    aq_scale: float
        Scale parameter of the unsaturated parametrization of the underlying
        aquifer [m].
    aq_shape: float
        Shape parameter of the unsaturated parametrization of the underlying
        aquifer [-].
    aq_para: str
        Name of the unsaturated parametrization of the underlying aquifer.
        Either vGM (van Genuchten - Mualem) or BCB (Brooks - Corey - Burdine).
    """
    
    if aq_para == 'vGM':
        q = q_approx_full_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape)
    elif aq_para == 'BCB':
        q = q_approx_full_BCB(stage, cl_cond, cl_th, aq_cond, aq_scale,
                              aq_shape)
        
    return q

def Phi(psi, psi_g, n):
    m = 1 - 1/n
    u = (psi / psi_g)**n
    f = 1 - u * (1 + u)**(-m)
    return (n * u / (1 + u)) * (m/2 + 2 * (1 + u)**(-m) * (1 + (1 - m) * u) / f)

def q_approx_full_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape):
    """
    Approximate seepage at full disconnection for the vGM parametrization.
    """
    
    q0 = q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    hc = cl_th * (aq_cond / cl_cond - 1)
    psi = cl_th * (q0 / cl_cond - 1)
    b = Phi(psi, aq_scale, aq_shape)
    s = -cl_cond / cl_th * (q0 - cl_cond + 1e-20) / ((1 + b) * q0 - cl_cond + 1e-20)
    hstar =  (q0 - cl_cond) / (s * hc + q0 - cl_cond) * hc
    q = q0 + (cl_cond / cl_th - s * hstar / (stage - hstar + 1e-20)) * stage

    return q


# def q_approx_full_vGM(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape):
#     """
#     Approximate seepage at full disconnection for the vGM parametrization.
#     """
    
#     q0 = q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

#     b = 0.5 * (5 * aq_shape - 1)
#     hc = cl_th * (aq_cond / cl_cond - 1)
#     x = np.min([0.99 * aq_cond, (1 + b) * q0], axis=0)
#     s = -cl_cond / cl_th * (q0 - cl_cond) / (x - cl_cond)
#     hstar =  (q0 - cl_cond) / (s * hc + q0 - cl_cond) * hc
#     q = q0 + (cl_cond / cl_th - s * hstar / (stage - hstar)) * stage

#     print(hstar)

#     return q

def q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape, 
                       C_NS_1=-0.3850, C_NS_2=0.2056, C_NS_3=0.5818,
                       C_SH_1=0.4633, C_SH_2=0.5396, C_NSH_1=1.05, C_NSH_2=0.25,
                       C_NSH_3=6):
    """
    Approximate seepage at full disconnection and zero ponding for the vGM
    parametrization.
    """

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

    return np.max([q0, cl_cond], axis=0)

def q_approx_full_BCB(stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape):
    """
    Approximate seepage at full disconnection for the BCB parametrization.
    """

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
    """
    Approximate seepage at full disconnection and zero ponding for the BCB
    parametrization.
    """

    b = 2 + 3 * aq_shape
    B = 1

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    return np.min([aq_cond, q0_sh], axis=0)

def q0_negl_to_soft(aq_cond, x, xi, a_ns):
    """Transition equation from negligible to soft"""
    return aq_cond * (1 + x**a_ns)**(-xi/a_ns)

def q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh):
    """Transition equation from soft to hard"""
    return cl_cond * (1 + (x_sh / x)**a_sh)**(xi / a_sh)

def q_modflow(stage: float, cl_cond: float, cl_th: float):
    """
    Approximate seepage through a clogging layer at full disconnection according
    to MODFLOW

    Parameters
    ----------
    stage: float
        Water depth in the stream [m].
    cl_cond: float
        Hydraulic conductivity of the clogging layer [m/s].
    cl_th: float
        Thickness of the clogging layer [m].
    """

    q = cl_cond * (1 + stage / cl_th)

    return q

def q0_asymptote_negl(cl_cond: float,  cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    """Negligible asymptotic solution"""
    q0 = aq_cond

    return q0

def q0_asymptote_soft(cl_cond: float, cl_th: float, aq_cond: float,
             aq_scale: float, aq_shape: float, aq_para: str):
    """Soft asymptotic solution"""
    
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
    """Hard asymptotic solution"""
    
    q0 = cl_cond
    
    return q0