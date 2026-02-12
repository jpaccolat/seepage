#!/usr/bin/env python

"""First experiment.

In this module, the infiltrability space is scanned to quantify
(1) the convergence toward the predicted soft regime power law,
(2) the size of the transition intervals and its dependence on the shape
parameter and the conductivity ratio.

A regular grid is defined from the constants given below. It spans the shape 
parameter, conductivity ratio and x-ratio. Ka and aq_scale are set to one
without loss of generality. 

The exact and asymptotic infiltrabilities of each grid point are saved in
rates.csv

The maximum log-log slope and the transition interval sizes are saved in
sl_it.csv
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
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Internal imports
from rate import q_exact_dis
from rate import q0_asymptote_negl
from rate import q0_asymptote_soft
from rate import q0_asymptote_hard
from utils import get_Dless_parameters

q_exact_dis = np.vectorize(q_exact_dis)

####################
# Constants        #
####################

# aquifer shape parameter
SCA_b = 'lin'
MIN_b = 2.1
MAX_b = 14
NUM_b = 50

# dimensionless x ratio
SCA_x = 'log'
MIN_x = -3
MAX_x = 12
NUM_x = 400

# hydraulic conductivity ratio
SCA_cond_ratio = 'log'
MIN_cond_ratio = 1
MAX_cond_ratio = 6
NUM_cond_ratio = 6

####################
# Functions        #
####################

def compute_rates(args):
    """
    Compute asymptotic and exact infiltrabilities (zero ponding rate) over
    a regular grid (defined from the module constants).

    Return a dataframe with the infiltrability values.
    """

    # intervals to scan
    b = np.linspace(MIN_b, MAX_b, NUM_b)
    x = np.logspace(MIN_x, MAX_x, NUM_x)
    cond_ratio = np.logspace(MIN_cond_ratio, MAX_cond_ratio, NUM_cond_ratio)

    # flatten the grid
    grids = np.meshgrid(b, x, cond_ratio, indexing='ij')
    b, x, cond_ratio = tuple(grid.flatten() for grid in grids)

    # set inputs
    aq_cond = 1.
    aq_scale = 1.

    if args.aq_para == 'vGM':
        aq_shape = (2 * b + 1) / 5
        B = (1 - 1 / aq_shape)**2
    elif args.aq_para == 'BCB':
        aq_shape = (b - 2) / 3
        B = 1
    else:
        print('Invalid parametrization. Use vGM or BCB.')

    cl_cond = aq_cond / cond_ratio
    cl_th = B**(1/b) * aq_scale / cond_ratio * x

    df = pd.DataFrame(index=range(NUM_b * NUM_x * NUM_cond_ratio),
                      columns=['stage', 'cl_cond', 'cl_th', 'aq_cond',
                               'aq_scale', 'aq_shape', 'aq_para'])
    df['stage'] = 0.
    df['cl_cond'] = cl_cond
    df['cl_th'] = cl_th
    df['aq_cond'] = aq_cond
    df['aq_scale'] = aq_scale
    df['aq_shape'] = aq_shape
    df['aq_para'] = args.aq_para

    # append dataframe with dimensionless parameters
    df['b'], df['B'], df['xi'], df['x'], df['xsh'] = get_Dless_parameters(
                                    df['cl_cond'], df['cl_th'], df['aq_cond'],
                                    df['aq_scale'], df['aq_shape'],
                                    args.aq_para
                                    )

    # compute rates
    df['q_exact_dis'] = q_exact_dis(0., cl_cond, cl_th, aq_cond, aq_scale,
                                    aq_shape, args.aq_para)
    df['q0_asymptote_negl'] = q0_asymptote_negl(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape,
                                                args.aq_para)
    df['q0_asymptote_soft'] = q0_asymptote_soft(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape,
                                                args.aq_para)
    df['q0_asymptote_hard'] = q0_asymptote_hard(cl_cond, cl_th, aq_cond,
                                                aq_scale, aq_shape,
                                                args.aq_para)
    
    return df

def compute_max_slopes(args, df_rates):
    """
    Compute the maximum slope between log10(x) and log10(q0), where q0 is the 
    exact infiltrability stored in *df_rates*, for fixed conductivity ratio and
    shape parameter.

    Return a dataframe with the max slopes.
    """

    
    df = pd.DataFrame(index=range(NUM_b * NUM_cond_ratio),
                      columns=['cond_ratio', 'b', 'max_slope'])
    
    aq_cond = 1.
    cl_cond = df_rates['cl_cond'].unique()
    b = df_rates['b'].unique()

    for i, (v1, v2) in enumerate(product(cl_cond, b)):
        df_cut = df_rates[(df_rates['cl_cond'] == v1) & (df_rates['b'] == v2)]
        grad = np.gradient(np.log10(df_cut['q_exact_dis'] / df_cut['aq_cond']),
                            np.log10(df_cut['x']))
        max_slope = -np.nanmin(grad)
        df.loc[i] = [aq_cond / v1, v2, max_slope]
            
    return df

def get_negl_soft_interval(df_rates, v1, v2, eps):
    """
    Compute the negligible to soft transition interval for fixed conductivity
    ratio and shape parameter.
    """

    df_cut = df_rates[(df_rates['cl_cond'] == v1) & (df_rates['b'] == v2)]

    # initial guess
    xns = 1 

    # relative error of negl. regime asymptote
    en = (df_cut['q0_asymptote_negl'] - df_cut['q_exact_dis']) / df_cut['q_exact_dis']
    en[en < 0] = np.nan # discard opposite hierachy (outside of domain)
    fn = interp1d(np.log10(df_cut['x']), en - eps, fill_value='extrapolate')
    xn = 10**fsolve(fn, np.log10(xns))[0]

    # relative error of soft regime asymptote
    es = (df_cut['q0_asymptote_soft'] - df_cut['q_exact_dis']) / df_cut['q_exact_dis']
    es[es < 0] = np.nan # discard opposite hierachy (outside of domain)
    fs = interp1d(np.log10(df_cut['x']), es - eps, fill_value='extrapolate')
    xs = 10**fsolve(fs, np.log10(xns))[0]

    return np.log10(xs / xn)

def get_soft_hard_interval(df_rates, v1, v2, eps):
    """
    Compute the soft to hard transition interval for fixed conductivity ratio
    and shape parameter.
    """

    df_cut = df_rates[(df_rates['cl_cond'] == v1) & (df_rates['b'] == v2)]

    # initial guess
    xsh = df_cut['xsh'].values[0]

    # relative error of negl. regime asymptote
    es = (df_cut['q_exact_dis'] - df_cut['q0_asymptote_soft']) / df_cut['q_exact_dis']
    es[es < 0] = np.nan # discard opposite hierachy (outside of domain)
    fs = interp1d(np.log10(df_cut['x']), es - eps, fill_value='extrapolate')
    xs = 10**fsolve(fs, np.log10(xsh))[0]

    # relative error of soft regime asymptote
    eh = (df_cut['q_exact_dis'] - df_cut['q0_asymptote_hard']) / df_cut['q_exact_dis']
    eh[eh < 0] = np.nan # discard opposite hierachy (outside of domain)
    fh = interp1d(np.log10(df_cut['x']), eh - eps, fill_value='extrapolate')
    xh = 10**fsolve(fh, np.log10(xsh))[0]

    return np.log10(xh / xs)

def compute_transition_intervals(args, df_rates):
    """
    Compute the transition intervals, delta_ns (btw, negl and soft asymp.
    solutions) and delta_sh (btw. soft and hard asymp. solutions). The intervals
    are defined as the log10 of the ratio btw. the left and right x-values where
    the relative error is *eps_interval* (default is 10%).
    
    Return a dataframe with the intervals.
    """

    df = pd.DataFrame(index=range(NUM_b * NUM_cond_ratio),
                      columns=['cond_ratio', 'b', 'delta_ns', 'delta_sh'])

    aq_cond = 1.
    cl_cond = df_rates['cl_cond'].unique()
    b = df_rates['b'].unique()

    for i, (v1, v2) in enumerate(product(cl_cond, b)):
        delta_ns = get_negl_soft_interval(df_rates, v1, v2, args.eps_interval)
        delta_sh = get_soft_hard_interval(df_rates, v1, v2, args.eps_interval)
        df.loc[i] = [aq_cond / v1, v2, delta_ns, delta_sh]

    return df

def run(args):
    """Run experiment 1."""

    df_rates = compute_rates(args)
    df_rates.to_csv(args.path / 'rates.csv', sep=',', index=True, header=True)
    print(f'Seepage rates stored in {args.path / 'rates.csv'}')

    df_slopes = compute_max_slopes(args, df_rates)
    df_intervals = compute_transition_intervals(args, df_rates)
    df_sl_it = pd.merge(df_slopes, df_intervals[['delta_ns', 'delta_sh']],
                        left_index=True, right_index=True, how='outer')
    df_sl_it.to_csv(args.path / 'sl_it.csv', sep=',', index=True, header=True)
    print(f'Slopes and intervals stored in {args.path / 'sl_it.csv'}')


def dump_intervals(args):
    """Save module constants which define the regular grid."""

    d = {            
        'b': [SCA_b, MIN_b, MAX_b, NUM_b],
        'x': [SCA_x, MIN_x, MAX_x, NUM_x],
        'cond_ratio': [SCA_cond_ratio, MIN_cond_ratio, MAX_cond_ratio,
                       NUM_cond_ratio]
    }   
    json.dump(d, open(args.path / 'intervals.txt', 'w'))

def main():

    parser = argparse.ArgumentParser(
                        prog='exp1',
                        description='Compute seepage rate over the whole' \
                        'infiltrability space')

    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--eps_interval', type=float, default=0.1,
                        help='Relative error defining the transition intervals')
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
            for f in args.path.iterdir(): f.unlink()
            print(f'Output folder {args.path} emptied.')
            pass
        else: raise

    # save constants
    dump_intervals(args)

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()