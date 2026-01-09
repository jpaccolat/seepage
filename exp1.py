#!/usr/bin/env python

"""First experiment.

In this module, the infiltrability space is scanned to quantify
(1) the convergence toward the predicted soft regime power law,
(2) the size of the transition intervals and its dependence on the shape
parameter and the conductivity ratio.
"""

####################
# Libraries        #
####################

# Standard imports
import pathlib
import argparse
import json
import time

# Third party imports
import numpy as np
import pandas as pd

# Internal imports
from rate import q0_exact
from rate import q0_asymptote_negl
from rate import q0_asymptote_soft
from rate import q0_asymptote_hard

q0_exact = np.vectorize(q0_exact)

####################
# Constants        #
####################

SCA_b = 'lin'
MIN_b = 2.1
MAX_b = 14
NUM_b = 20

SCA_x = 'log'
MIN_x = -3
MAX_x = 12
NUM_x = 20

SCA_cond_ratio = 'log'
MIN_cond_ratio = 1
MAX_cond_ratio = 6
NUM_cond_ratio = 6

####################
# Functions        #
####################

def dump_intervals(args):

    d = {            
        'b': [SCA_b, MIN_b, MAX_b, NUM_b],
        'x': [SCA_x, MIN_x, MAX_x, NUM_x],
        'cond_ratio': [SCA_cond_ratio, MIN_cond_ratio, MAX_cond_ratio,
                       NUM_cond_ratio]
    }   
    json.dump(d, open(args.path / 'intervals.txt', 'w'))

def get_grid_coordinates(*intervals):

    grids = np.meshgrid(*intervals, indexing='ij')
    
    return tuple(grid.flatten() for grid in grids)

def run(args):

    # intervals to scan
    b = np.linspace(MIN_b, MAX_b, NUM_b)
    x = np.logspace(MIN_x, MAX_x, NUM_x)
    cond_ratio = np.logspace(MIN_cond_ratio, MAX_cond_ratio, NUM_cond_ratio)

    # flatten the grid
    b, x, cond_ratio = get_grid_coordinates(b, x, cond_ratio)

    # set inputs
    aq_cond = 1
    aq_scale = 1

    if args.aq_para == 'vGM':
        aq_shape = (2 * b + 1) / 5
        B = (1 - 1 / aq_shape)**2
    elif args.aq_para == 'BCB':
        aq_shape = (b - 2) / 3
        B = 1
    else:
        print('Invalid parametrization. Use vGM or BCB.')

    cl_cond = aq_cond / cond_ratio
    cl_th = B**(1/b) * aq_scale / cond_ratio

    df = pd.DataFrame(index=range(len(b)),
                      columns=['stage', 'cl_cond', 'cl_th', 'aq_cond',
                               'aq_scale', 'aq_shape', 'aq_para'])
    df['stage'] = 0.
    df['cl_cond'] = cl_cond
    df['cl_th'] = cl_th
    df['aq_cond'] = aq_cond
    df['aq_scale'] = aq_scale
    df['aq_shape'] = aq_shape
    df['aq_para'] = args.aq_para

    # compute rates

    t1 = time.perf_counter()
    df['q0_exact'] = q0_exact(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                              args.aq_para)
    t2 = time.perf_counter()
    df['q0_asymptote_negl'] = q0_asymptote_negl(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape, args.aq_para)
    t3 = time.perf_counter()
    df['q0_asymptote_soft'] = q0_asymptote_soft(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape, args.aq_para)
    t4 = time.perf_counter()
    df['q0_asymptote_hard'] = q0_asymptote_hard(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape, args.aq_para)
    t5 = time.perf_counter()
    
    print(f'{t2-t1:.3f}, {t3-t2:.5f}, {t4-t3:.5f}, {t5-t4:.5f}')

    # save data
    df.to_csv(args.path / 'rates.csv', sep=',', index=True, header=True)

def main():

    parser = argparse.ArgumentParser(
                        prog='exp1',
                        description='Compute seepage rate over the whole' \
                        'infiltrability space')

    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--output', type=str, help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp1_{args.aq_para}'

    # handle output directory
    args.path = pathlib.Path(args.output)
    try:
        args.path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if args.clean:
            print(f'Output folder {args.path} emptied.')
            pass
        else: raise

    # save constants
    dump_intervals(args)

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()