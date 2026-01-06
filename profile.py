#!/usr/bin/env python

"""Profile module.

This module defines seepage profiles which consists of:
 - cl_cond (clogging conductivity)
 - cl_th (clogging thickness)
 - aq_cond (aquifer conductivity)
 - aq_shape (aquifer unsaturated shape parameter)
 - aq_scale (aquifer unsaturated scale parameter)
 - aq_para (aquifer unsaturated parametrization function)
 - stage (SW stage)

The module contains functions to obtaine profile properties, e.g., scaling exponent, clogging regime, x-value,...

The module contains functions to generate profiles randomly or on a regular grid.
"""