#!/usr/bin/env python

"""Third experiment.

Compute exact and approximate infiltrabilities.
"""

####################
# Libraries        #
####################

# Standard imports
import pathlib
import argparse
from time import perf_counter

#import warnings
#warnings.filterwarnings('ignore')

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal imports
from rate import q_exact
from rate import q_approx
from rate import q_exact_full
from rate import q_approx_full
from rate import q_modflow
import rose

####################
# Constants        #
####################

MIN_LOG_cl_cond = -8
MAX_LOG_cl_cond = -6
MIN_cl_th = 0
MAX_cl_th = 3
MIN_stage = 0
MAX_stage = 5
MIN_depth = 0
MAX_depth = 1

####################
# Functions        #
###################

def draw_samples(args, N):

    bounds = {
        'K': [0, np.inf],
        'hg': [0, np.inf],
        'n': [args.nmin, np.inf]
    }

    aq = rose.sample_soils(args.texture, N, bounds=bounds)
    aq_cond = aq['K']
    aq_scale = aq['hg']
    aq_shape = aq['n']

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

    return depth, stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape,

def run(args):
    """Run experiment 4."""

    df = pd.DataFrame(columns=['mf mean', 'mf std', 'ap full mean',
                               'ap full std', 'ex full mean', 'ex full std',
                               'ap mean', 'ap std', 'ex mean', 'ex std'])
    
    for i in tqdm(range(1, args.n+1)):
        N = args.base**i
        n_sample = 3 * args.base**(args.n-i)

        dt_mf = []
        dt_ap_full = []
        dt_ex_full = []
        dt_ap = []
        for k in range(n_sample):
            parameters = draw_samples(args, N)

            t1 = perf_counter()
            _ = q_modflow(*parameters[:3])
            t2 = perf_counter()
            _ = q_approx_full(*parameters[1:], args.aq_para)
            t3 = perf_counter()
            _ = q_exact_full(*parameters[1:], args.aq_para)
            t4 = perf_counter()
            _ = q_approx(*parameters, args.aq_para)
            t5 = perf_counter()

            dt_mf.append((t2 - t1) / N)
            dt_ap_full.append((t3 - t2) / N)
            dt_ex_full.append((t4 - t3) / N)
            dt_ap.append((t5 - t4) / N)

        df.loc[i, 'mf mean'] = np.mean(dt_mf)
        df.loc[i, 'mf std'] = np.std(dt_mf)
        df.loc[i, 'ap full mean'] = np.mean(dt_ap_full)
        df.loc[i, 'ap full std'] = np.std(dt_ap_full)
        df.loc[i, 'ex full mean'] = np.mean(dt_ex_full)
        df.loc[i, 'ex full std'] = np.std(dt_ex_full)
        df.loc[i, 'ap mean'] = np.mean(dt_ap)
        df.loc[i, 'ap std'] = np.std(dt_ap)
        df.loc[i, 'ex mean'] = np.nan
        df.loc[i, 'ex std'] = np.nan

    if args.n_BVP > 0:
        for i in tqdm(range(1, args.n_BVP+1)):
            N = args.base**i
            n_sample = 3 * args.base**(args.n_BVP-i)

            dt_ex = []
            for k in tqdm(range(n_sample)):
                parameters = draw_samples(args, N)

                t1 = perf_counter()
                _ = q_exact(*parameters, args.aq_para)
                t2 = perf_counter()

                dt_ex.append((t2 - t1) / N)

            df.loc[i, 'ex mean'] = np.mean(dt_ex)
            df.loc[i, 'ex std'] = np.std(dt_ex)

    df.to_csv(args.path / 'time.csv', sep=',', index=True, header=True)
    print(f'Data stored in {args.path / 'time.csv'}')

def main():

    parser = argparse.ArgumentParser(prog='exp5', description='')
    parser.add_argument('--aq_para', type=str, required=True,
                        help='vGM or BCB')
    parser.add_argument('--texture', type=str, required=True,
                        help='SAND or SAND_LOAM')
    parser.add_argument('--nmin', type=float, default=1.1,
                        help='Minimal value of the vGM shape parameter')
    parser.add_argument('--base', type=int, default=3)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--n_BVP', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp6_{args.aq_para}_{args.texture}'

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