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