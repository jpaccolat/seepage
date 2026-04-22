#!/usr/bin/env python

"""Fourth experiment.

Compute seepage relative errors for 
- the MODFLOW formula,
- the novel approximate expression at full disconnection (q_approx_full), and
- the novel approximate expression for finite water table depth (q_approx); only
the maximal relative error is retained.

The user defines the unsaturated parametrization: vGM or BCB.

The user defines the aquifer texture: SAND or SAND_LOAM.

The user defines the upper and lower bounds of the other parameters. Stage and
clogging thickness are drawn from a uniform distribution, while clogging
conductivity is drawn from a log-uniform distribution.
"""

####################
# Libraries        #
####################

# Standard imports
import sys
import pathlib
import argparse

import warnings
warnings.filterwarnings('ignore')

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal imports
sys.path.append('..')
from rate import q_exact
from rate import q_approx
from rate import q_exact_full
from rate import q_approx_full
from rate import q_modflow
import rose

####################
# Functions        #
###################

def draw_samples(args):
    """
    Generate random arrays of stage, cl_cond, cl_th, aq_cond, aq_scale and
    aq_shape.
    """

    stage = np.random.uniform(low=args.MIN_stage, high=args.MAX_stage,
                              size=args.n_sample)
    cl_cond = 10**np.random.uniform(low=args.MIN_LOG_cl_cond,
                                    high=args.MAX_LOG_cl_cond,
                                    size=args.n_sample)
    cl_th = np.random.uniform(low=args.MIN_cl_th, high=args.MAX_cl_th,
                              size=args.n_sample)

    # aquifer properties
    bounds = {
        'K': [0, np.inf],
        'hg': [0, np.inf],
        'n': [args.nmin, np.inf]
    }

    aq = rose.sample_from_class(args.texture, args.n_sample, bounds=bounds)
    aq_cond = aq['K']
    aq_scale = aq['hg']
    aq_shape = aq['n']

    if args.aq_para == 'BCB':
        b = 0.5 * (5 * aq_shape - 1)
        B = (1 - 1 / aq_shape)**2
        aq_scale *= B**(1/b)
        aq_shape = (b - 2) / 3
    else: assert args.aq_para == 'vGM', 'Bad parametrization. Use vGM or BCB.'

    return stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape

def run(args):
    """Compute seepage relative errors and save them in a csv file."""

    # draw parameters
    stage, cl_cond, cl_th, aq_cond, aq_scale, aq_shape = draw_samples(args)

    # setup dataframe
    df = pd.DataFrame(columns=['stage', 'cl_cond', 'cl_th', 'aq_cond',
                               'aq_scale', 'aq_shape', 'aq_para',
                               'rel_err_mf', 'rel_err_dis', 'rel_err_max'
                               'unsaturated', 'clogged', 'van_cap_zone',
                               'increasing'])
    df['stage'] = stage
    df['cl_cond'] = cl_cond
    df['cl_th'] = cl_th
    df['aq_cond'] = aq_cond
    df['aq_scale'] = aq_scale
    df['aq_shape'] = aq_shape
    df['aq_para'] = args.aq_para
    df['rel_err_mf'] = np.nan
    df['rel_err_dis'] = np.nan
    df['rel_err_max'] = np.nan
    df['van_cap_zone'] = False
    df['increasing'] = True

    # check unsaturated condition
    hc = cl_th * (aq_cond / cl_cond - 1)
    if args.aq_para == 'BCB': hc -= aq_scale
    df['unsaturated'] = stage < hc

    # check clogging condition
    df['clogged'] = cl_cond < aq_cond
    
    # compute fully disconnected seepage
    idx = np.logical_and(df['unsaturated'], df['clogged'])
    print(f'{idx.sum()} valid configurations')

    q_mf = q_modflow(stage[idx], cl_cond[idx], cl_th[idx])
    q_ap_d = q_approx_full(stage[idx], cl_cond[idx], cl_th[idx], aq_cond[idx],
                           aq_scale[idx], aq_shape[idx], args.aq_para)
    q_ex_d = q_exact_full(stage[idx], cl_cond[idx], cl_th[idx], aq_cond[idx],
                          aq_scale[idx], aq_shape[idx], args.aq_para)
    df.loc[idx, 'rel_err_mf'] = (q_mf - q_ex_d) / q_ex_d
    df.loc[idx, 'rel_err_dis'] = (q_ap_d - q_ex_d) / q_ex_d
        
    # compute transionaly diconnected seepage
    depth_dis = (cl_th[idx] * (q_ex_d / cl_cond[idx] - 1) - stage[idx]) \
                / (1 - q_ex_d / aq_cond[idx])

    print('Start searching max errors')
    for i in tqdm(df.loc[idx].index):

        # assert depth to disconnection is well defined
        if np.isnan(depth_dis[i]):
            print('Unresolved disconnected depth')
            print(stage[i], cl_cond[i], cl_th[i], aq_cond[i], aq_scale[i],
                  aq_shape[i])
            continue
    
        # assert depth to disconnection is not vanishing (or negative)
        if depth_dis[i] < 5e-2:
            df.loc[i, 'van_cap_zone'] = True
            df.loc[i, 'rel_err_max'] = df.loc[i, 'rel_err_dis']
            continue

        # scan exact solution over a sparse interval (slow to compute)
        n_sparse = min(10, int(depth_dis[i] / 2e-2) + 1)
        w_sparse = np.linspace(0, 2 * depth_dis[i], n_sparse)
        q_ex = q_exact(w_sparse, np.ones_like(w_sparse) * stage[i], cl_cond[i],
                       cl_th[i], aq_cond[i], aq_scale[i], aq_shape[i],
                       args.aq_para, max_nodes=100)
        # assert solution is correct
        if sorted(q_ex) != list(q_ex): 
            df.loc[i, 'increasing'] = False
            continue

        # scan approximate solution over a dense interval
        n_dense = 10 * n_sparse
        w_dense = np.linspace(0, 2 * depth_dis[i], n_dense)
        q_ap = q_approx(w_dense, np.ones_like(w_dense) * stage[i], cl_cond[i],
                        cl_th[i], aq_cond[i], aq_scale[i], aq_shape[i],
                        args.aq_para)
        
        # compute relative error
        q_ex = np.interp(w_dense, w_sparse, q_ex)
        df.loc[i, 'rel_err_max'] = max((q_ap - q_ex) / q_ex)

    # save data
    df.to_csv(args.path / 'rel_error.csv', sep=',', index=True, header=True)
    print(f'Data stored in {args.path / 'rel_error.csv'}')

def main():

    parser = argparse.ArgumentParser(prog='exp4', description='')
    parser.add_argument('--aq_para', type=str, required=True,
                        help='vGM or BCB')
    parser.add_argument('--texture', type=str, required=True,
                        help='SAND or SAND_LOAM')
    parser.add_argument('--n_sample', type=int, required=True,
                        help='number of samples')
    parser.add_argument('--nmin', type=float, default=1.1,
                        help='Minimal value of the vGM shape parameter')
    parser.add_argument('--MIN_cl_th', type=float, default=0)
    parser.add_argument('--MAX_cl_th', type=float, default=2)
    parser.add_argument('--MIN_stage', type=float, default=0)
    parser.add_argument('--MAX_stage', type=float, default=6)
    parser.add_argument('--MIN_LOG_cl_cond', type=float, default=-8)
    parser.add_argument('--MAX_LOG_cl_cond', type=float, default=-6)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp4_{args.aq_para}_{args.texture}'

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