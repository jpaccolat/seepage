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
import rosetta

q_exact = np.vectorize(q_exact)
q_approx = np.vectorize(q_approx)
q_exact_dis = np.vectorize(q_exact_dis)
q_approx_dis = np.vectorize(q_approx_dis)

####################
# Constants        #
####################

LIST_stage = [0., 0.2, 1.0, 2.0]
LIST_cl_cond = [1e-8, 1e-6]
LIST_cl_th = [0.1, 1.]

####################
# Functions        #
###################

def draw_samples(args):

    if args.texture == 'SAND':
        aq = rosetta.get_sand(args.n_sample)
    elif args.texture == 'SILT':
        aq = rosetta.get_silt(args.n_sample)
    else:
        print('Texture must be SAND or SILT.')

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

    aq_cond, aq_scale, aq_shape = draw_samples(args)

    df = pd.DataFrame(columns=['stage', 'cl_cond', 'cl_th', 'aq_cond',
                               'aq_scale', 'aq_shape', 'aq_para', 'unsaturated',
                               'clogged', 'rel_err_dis', 'rel_err_mf',
                               'rel_err_max'])
    
    for v1, v2, v3 in tqdm(product(LIST_stage, LIST_cl_cond, LIST_cl_th)):

        df_ = pd.DataFrame(index=range(args.n_sample))
        df_['aq_cond'] = aq_cond
        df_['aq_scale'] = aq_scale
        df_['aq_shape'] = aq_shape
        df_['aq_para'] = args.aq_para
        df_['stage'] = v1
        df_['cl_cond'] = v2
        df_['cl_th'] = v3

        # check unsaturated condition
        hc = v3 * (aq_cond / v2 - 1)
        if args.aq_para == 'BCB': hc -= aq_scale
        df_['unsaturated'] = v1 < hc

        # check clogging condition
        df_['clogged'] = v2 < aq_cond
        
        idx = np.logical_and(df_['unsaturated'], df_['clogged'])
    
        # compute rel. error when fully disconnected
        q_ex_d = np.full(args.n_sample, np.nan, dtype=float)
        q_ex_d[idx] = q_exact_dis(v1, v2, v3, aq_cond[idx], aq_scale[idx],
                                  aq_shape[idx], args.aq_para)
        
        q_ap_d = np.full(args.n_sample, np.nan, dtype=float)
        q_ap_d[idx] = q_approx_dis(v1, v2, v3, aq_cond[idx], aq_scale[idx],
                                   aq_shape[idx], args.aq_para)
        q_mf = q_modflow(v1, v2, v3)

        df_['rel_err_dis'] = (q_ap_d - q_ex_d) / q_ex_d
        df_['rel_err_mf'] = (q_mf - q_ex_d) / q_ex_d

        # compute maximal rel. error (for a finite WT depth)
        depth_dis = v3 * ((q_ex_d / v2 - 1) - v1) / (1 - q_ex_d / aq_cond)
        for i in range(args.n_sample):

            if np.isnan(depth_dis[i]):
                err = np.nan
                continue

            w_sparse = np.linspace(0, 2 * depth_dis[i], 10)
            w_dense = np.linspace(0, 2 * depth_dis[i], 100)

            q_ex = q_exact(v1, w_sparse, v2, v3, aq_cond[i], aq_scale[i],
                           aq_shape[i], args.aq_para, max_nodes=100)
            q_ex = np.interp(w_dense, w_sparse, q_ex)
            
            q_ap = q_approx(v1, w_dense, v2, v3, aq_cond[i], aq_scale[i],
                            aq_shape[i], args.aq_para)
            err = max((q_ap - q_ex) / q_ex)
            
            df_.loc[i, 'rel_err_max'] = err

        df = pd.concat([df, df_], ignore_index=True)

    df.to_csv(args.path / 'rel_error.csv', sep=',', index=True, header=True)
    print(f'Seepage rates stored in {args.path / 'rel_error.csv'}')

def dump_constants(args):
    """Save module constants which define the regular grid."""

    d = {         
        'stage': LIST_stage,   
        'cl_th': LIST_cl_th,
        'cl_cond': LIST_cl_cond
    }   
    json.dump(d, open(args.path / 'constants.txt', 'w'))

def main():

    parser = argparse.ArgumentParser(
                        prog='exp4',
                        description='')
    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--texture', type=str, required=True,
                        help='SAND or SILT')
    parser.add_argument('--n_sample', type=int, required=True,
                        help='number of samples')
    parser.add_argument('--output', type=str, help='Path to directory')
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

    # save constants
    dump_constants(args)

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()