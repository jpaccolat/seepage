#!/usr/bin/env python

"""Profile module.

This module defines seepage profiles which consists of:
 - stage (SW stage)
 - cl_cond (clogging conductivity)
 - cl_th (clogging thickness)
 - aq_cond (aquifer conductivity)
 - aq_shape (aquifer unsaturated shape parameter)
 - aq_scale (aquifer unsaturated scale parameter)
 - aq_para (aquifer unsaturated parametrization function)

The module contains functions to obtaine profile properties, e.g., scaling exponent, clogging regime, x-value,...

The module contains functions to generate profiles randomly or on a regular grid.
"""

def get_generalized_shape_parameters(aq_shape, aq_para):

    if aq_para == 'vGM':
        b = 0.5 * (5 * aq_shape - 1)
        B = (1 - 1 / aq_shape)**2
    
    elif aq_para == 'BCB':
        b = 2 + 3 * aq_shape
        B = 1

    return b, B

def get_Dless_parameters(cl_cond, cl_th, aq_cond, aq_scale, aq_shape, aq_para):

    b, B = get_generalized_shape_parameters(aq_shape, aq_para)
    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    return b, B, xi, x, x_sh