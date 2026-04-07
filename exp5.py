#!/usr/bin/env python

"""Fifth experiment.

Measure the time complexity to compute seepage for the different formulas, i.e.
exact and approximate, with infinite or finite water table depth. 

Array of incresing sizes are considered. From base to base**n, where base and n
are user defined. 

The exact solution with finite water table depth requires solving a BVP, which 
requires much more time. It is thus treated apart, with a smaller exponent:
n_BVP. If the latter is not specified, this solution is discarded.
"""

####################
# Libraries        #
####################

# Standard imports
import pathlib
import argparse
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

# Third party imports
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

# Internal imports
import rate
import rose

####################
# Constants        #
####################

MIN_LOG_cl_cond = -8
MAX_LOG_cl_cond = -6
MIN_cl_th = 0
MAX_cl_th = 2
MIN_stage = 0
MAX_stage = 6
MIN_depth = 0
MAX_depth = 1

####################
# Functions        #
####################

def draw_samples(args, N):
    """
    Generate random arrays of stage, cl_cond, cl_th, aq_cond, aq_scale and
    aq_shape.
    """

    bounds = {
        'K': [0, np.inf],
        'hg': [0, np.inf],
        'n': [args.nmin, np.inf]
    }

    aq = rose.sample_from_class(args.texture, N, bounds=bounds)
    aq_cond = aq['K'].values
    aq_scale = aq['hg'].values
    aq_shape = aq['n'].values

    if args.aq_para == 'BCB':
        b = 0.5 * (5 * aq_shape - 1)
        B = (1 - 1 / aq_shape)**2
        aq_scale *= B**(1/b)
        aq_shape = (b - 2) / 3
    else: assert args.aq_para == 'vGM', 'Bad parametrization. Use vGM or BCB.'

    cl_cond = 10**np.random.uniform(low=MIN_LOG_cl_cond, high=MAX_LOG_cl_cond,
                                    size=N)
    cl_th = np.random.uniform(low=MIN_cl_th, high=MAX_cl_th, size=N)
    stage = np.random.uniform(low=MIN_stage, high=MAX_stage, size=N)
    depth = np.random.uniform(low=MIN_depth, high=MAX_depth, size=N)

    return depth, stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape

def run(args):
    """Run experiment 4."""

    # init dataframes
    columns = ['mf', 'ap full st', 'ap full', 'ap st', 'ap', 'ex full']
    abs_m = pd.DataFrame(columns=columns)
    abs_s = pd.DataFrame(columns=columns)
    rel_m = pd.DataFrame(columns=columns)
    rel_s = pd.DataFrame(columns=columns)
    
    # set BLAS thread = 1 for benchmarking
    with threadpool_limits(limits=1, user_api="blas"):
        # increase array size from base to base**n
        # the computation time is averaged in a way that the total number of
        # requests is independent on the array size
        for i in tqdm(range(1, args.n+1)):
            N = args.base**i
            n_sample = 3 * args.base**(args.n-i)

            dt = {
                'mf': [],
                'ap full st': [],
                'ap full': [],
                'ap st': [],
                'ap': [],
                'ex full': []
            }
            for k in range(n_sample):
                p = draw_samples(args, N)

                if args.aq_para == 'vGM':
                    statics = rate.get_statics_vGM(*p[2:])
                else:
                    statics = rate.get_statics_BCB(*p[2:])

                t1 = perf_counter()
                _ = rate.q_modflow(*p[:3])
                t2 = perf_counter()
                _ = rate.q_approx_full(*p[1:4], statics=statics)
                t3 = perf_counter()
                _ = rate.q_approx_full(*p[1:4], aq_cond=p[4], aq_scale=p[5],
                                       aq_shape=p[6], aq_para=args.aq_para)
                t4 = perf_counter()
                _ = rate.q_approx(*p[:5], statics=statics)
                t5 = perf_counter()
                _ = rate.q_approx(*p[:5], aq_scale=p[5], aq_shape=p[6],
                                  aq_para=args.aq_para)
                t6 = perf_counter()
                _ = rate.q_exact_full(*p[1:], args.aq_para)
                t7 = perf_counter()

                dt['mf'].append(t2 - t1)
                dt['ap full st'].append(t3 - t2)
                dt['ap full'].append(t4 - t3)
                dt['ap st'].append(t5 - t4)
                dt['ap'].append(t6 - t5)
                dt['ex full'].append(t7 - t6)

            for key in columns:
                dt[key] = np.array(dt[key])

            for key in columns:
                abs_m.loc[i, key] = np.mean(dt[key])
                abs_s.loc[i, key] = np.std(dt[key])
                rel_m.loc[i, key] = np.mean(dt[key] / dt['ex full'])
                rel_s.loc[i, key] = np.std(dt[key] / dt['ex full'])

    # save data
    abs_m.to_csv(args.path / 'abs_m.csv', sep=',', index=True, header=True)
    abs_s.to_csv(args.path / 'abs_s.csv', sep=',', index=True, header=True)
    rel_m.to_csv(args.path / 'rel_m.csv', sep=',', index=True, header=True)
    rel_s.to_csv(args.path / 'rel_s.csv', sep=',', index=True, header=True)
    print(f'Data stored in {args.path}')

def main():

    parser = argparse.ArgumentParser(prog='exp5', description='')
    parser.add_argument('--aq_para', type=str, required=True,
                        help='vGM or BCB')
    parser.add_argument('--texture', type=str, required=True,
                        help='SAND or SAND_LOAM')
    parser.add_argument('--nmin', type=float, default=1.1,
                        help='Minimal value of the vGM shape parameter')
    parser.add_argument('--base', type=int, default=5)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp5_{args.aq_para}_{args.texture}'

    # handle output directory
    args.path = pathlib.Path(args.output)
    try:
        args.path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if args.clean:
            for f in args.path.iterdir(): f.unlink()
            print(f'Output folder {args.path} emptied.')
            pass
        else: 
            print(f'Output folder {args.path} is not empty.')
            if input('Do you want to continue: y/n \n') == 'y':
                pass
            else:
                return 0

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()