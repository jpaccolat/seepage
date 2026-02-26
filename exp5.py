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
import json
from itertools import product

import warnings
warnings.filterwarnings('ignore')

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal imports
from rate import q_exact
from rate import q_approx
from rate import q_exact_dis
from rate import q_approx_dis
from rate import q_modflow
from utils import timer
import rose

q_exact = np.vectorize(timer(q_exact))
q_approx = np.vectorize(timer(q_approx))
q_exact_dis = np.vectorize(timer(q_exact_dis))
q_approx_dis = np.vectorize(timer(q_approx_dis))
q_modflow = np.vectorize(timer(q_modflow))

####################
# Constants        #
####################

MIN_LOG_cl_cond = -8
MAX_LOG_cl_cond = -6
MIN_cl_th = 0
MAX_cl_th = 3
MIN_stage = 0
MAX_stage = 5

####################
# Functions        #
###################

def draw_samples(args):

    bounds = {
        'K': [0, np.inf],
        'hg': [0, np.inf],
        'n': [args.nmin, np.inf]
    }

    aq = rose.sample_soils(args.texture, args.n_sample, bounds=bounds)
    aq_cond = aq['K']
    aq_scale = aq['hg']
    aq_shape = aq['n']

    if args.aq_para == 'BCB':
        b = 0.5 * (5 * aq_shape - 1)
        B = (1 - 1 / aq_shape)**2
        aq_scale *= B**(1/b)
        aq_shape = (b - 2) / 3
    else: assert args.aq_para == 'vGM', 'Bad parametrization. Use vGM or BCB.'

    return aq_cond, aq_scale, aq_shape

def run(args):
    """Run experiment 4."""

    # draw parameters
    aq_cond, aq_scale, aq_shape = draw_samples(args)
    cl_cond = 10**np.random.uniform(low=MIN_LOG_cl_cond, high=MAX_LOG_cl_cond,
                                    size=args.n_sample)
    cl_th = np.random.uniform(low=MIN_cl_th, high=MAX_cl_th, size=args.n_sample)
    stage = np.random.uniform(low=MIN_stage, high=MAX_stage, size=args.n_sample)

    # setup dataframe
    df = pd.DataFrame(columns=['stage', 'cl_cond', 'cl_th', 'aq_cond',
                               'aq_scale', 'aq_shape', 'aq_para',
                               'unsaturated', 'clogged', 'van_cap_zone',
                               'dt_ap_dis', 'dt_ex_dis', 'dt_mf', 'dt_ex_mean',
                               'dt_ap_mean', 'rel_err_mf', 'rel_err_dis',
                               'rel_err_max'])
    df['stage'] = stage
    df['cl_cond'] = cl_cond
    df['cl_th'] = cl_th
    df['aq_cond'] = aq_cond
    df['aq_scale'] = aq_scale
    df['aq_shape'] = aq_shape
    df['aq_para'] = args.aq_para
    df['van_cap_zone'] = False
    df['dt_ap_dis'] = np.nan
    df['dt_ex_dis'] = np.nan
    df['dt_mf'] = np.nan
    df['dt_ex_mean'] = np.nan
    df['dt_ap_mean'] = np.nan
    df['rel_err_mf'] = np.nan
    df['rel_err_dis'] = np.nan
    df['rel_err_max'] = np.nan

    # check unsaturated condition
    hc = cl_th * (aq_cond / cl_cond - 1)
    if args.aq_para == 'BCB': hc -= aq_scale
    df['unsaturated'] = stage < hc

    # check clogging condition
    df['clogged'] = cl_cond < aq_cond
    
    # compute fully disconnected seepage
    idx = np.logical_and(df['unsaturated'], df['clogged'])
    print(f'{idx.sum()} valid configurations')

    q_mf, dt_mf = q_modflow(stage[idx], cl_cond[idx], cl_th[idx])
    q_ap_d, dt_ap_d = q_approx_dis(stage[idx], cl_cond[idx], cl_th[idx],
                                   aq_cond[idx], aq_scale[idx], aq_shape[idx],
                                   args.aq_para)
    q_ex_d, dt_ex_d = q_exact_dis(stage[idx], cl_cond[idx], cl_th[idx],
                                  aq_cond[idx], aq_scale[idx], aq_shape[idx],
                                  args.aq_para)
    rel_err_dis = (q_ap_d - q_ex_d) / q_ex_d
    rel_err_mf = (q_mf - q_ex_d) / q_ex_d

    df.loc[idx, 'rel_err_mf'] = rel_err_mf
    df.loc[idx, 'rel_err_dis'] = rel_err_dis
    df.loc[idx, 'dt_mf'] = dt_mf
    df.loc[idx, 'dt_ap_dis'] = dt_ap_d
    df.loc[idx, 'dt_ex_dis'] = dt_ex_d
        
    # compute transionaly diconnected seepage
    depth_dis = (cl_th[idx] * (q_ex_d / cl_cond[idx] - 1) - stage[idx]) \
                / (1 - q_ex_d / aq_cond[idx])
    depth_dis = depth_dis.values

    rel_err_max = np.full(shape=len(depth_dis), fill_value=np.nan)
    dt_ap_mean = np.full(shape=len(depth_dis), fill_value=np.nan)
    dt_ex_mean = np.full(shape=len(depth_dis), fill_value=np.nan)
    van_cap_zone = np.zeros(len(depth_dis), dtype=bool)

    print('Start searching max errors')
    for i in tqdm(range(len(depth_dis))):

        if np.isnan(depth_dis[i]):
            print('Unresolved disconnected depth')
            print(stage[idx][i], cl_cond[idx][i], cl_th[idx][i],
                  aq_cond[idx][i], aq_scale[idx][i], aq_shape[idx][i])
            continue
    
        if depth_dis[i] < 5e-2:
            van_cap_zone[i] = True
            rel_err_max[i] = rel_err_dis[i]
            continue

        n_sparse = min(10, int(depth_dis[i] / 2e-2) + 1)
        w_sparse = np.linspace(0, 2 * depth_dis[i], n_sparse)
        n_dense = 10 * n_sparse
        w_dense = np.linspace(0, 2 * depth_dis[i], n_dense)

        q_ex, dt_ex = q_exact(stage[idx][i], w_sparse, cl_cond[idx][i], cl_th[idx][i],
                       aq_cond[idx][i], aq_scale[idx][i], aq_shape[idx][i],
                       args.aq_para, max_nodes=100)
        q_ex = np.interp(w_dense, w_sparse, q_ex)

        q_ap, dt_ap = q_approx(stage[idx][i], w_dense, cl_cond[idx][i], cl_th[idx][i],
                        aq_cond[idx][i], aq_scale[idx][i], aq_shape[idx][i],
                        args.aq_para)
        
        rel_err_max[i] = max((q_ap - q_ex) / q_ex)
        dt_ap_mean[i] = dt_ap.mean()
        dt_ex_mean[i] = dt_ex.mean()

        print('Variability of trans. disc. comp. time:')
        print(dt_ap.std() / dt_ap.mean())
        print(dt_ex.std() / dt_ex.mean())

    df.loc[idx, 'rel_err_max'] = rel_err_max
    df.loc[idx, 'van_cap_zone'] = van_cap_zone
    df.loc[idx, 'dt_ex_mean'] = dt_ex_mean
    df.loc[idx, 'dt_ap_mean'] = dt_ap_mean

    df.to_csv(args.path / 'rel_error.csv', sep=',', index=True, header=True)
    print(f'Data stored in {args.path / 'rel_error.csv'}')

def main():

    parser = argparse.ArgumentParser(prog='exp5', description='')
    parser.add_argument('--aq_para', type=str, required=True,
                        help='vGM or BCB')
    parser.add_argument('--texture', type=str, required=True,
                        help='SAND or SAND_LOAM')
    parser.add_argument('--n_sample', type=int, required=True,
                        help='number of samples')
    parser.add_argument('--nmin', type=float, default=1.1,
                        help='Minimal value of the vGM shape parameter')
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