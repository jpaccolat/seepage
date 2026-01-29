#!/usr/bin/env python

"""Utilities."""

####################
# Libraries        #
####################

# Standard imports
import functools
import time

####################
# Functions        #
####################

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

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        #print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value, elapsed_time
    return wrapper_timer