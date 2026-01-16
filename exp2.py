#!/usr/bin/env python

"""Second experiment.

In this module, the parameters of the approximate formula are fitted.
"""

####################
# Libraries        #
####################

# Standard imports
import pathlib
import argparse
import json

# Third party imports
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

# Internal imports
from rate import q0_exact
from rate import q0_negl_to_soft
from rate import q0_soft_to_hard

q0_exact = np.vectorize(q0_exact)

####################
# Constants        #
####################

SCA_b = 'lin'
MIN_b = 2.1
MAX_b = 14
NUM_b = 50
FIX_cond_ratio = 1e12
DEL_LOG_x = 0.2

####################
# Functions        #
####################

def fit_a_sh(args):

    # interval to scan
    b = np.linspace(MIN_b, MAX_b, NUM_b)

    # convert to shape parameters
    if args.aq_para == 'vGM':
        aq_shape = (2 * b + 1) / 5
        B = (1 - 1 / aq_shape)**2
    elif args.aq_para == 'BCB':
        aq_shape = (b - 2) / 3
        B = np.ones_like(b)
    else:
        print('Invalid parametrization. Use vGM or BCB.')

    # set inputs
    aq_cond = 1.
    aq_scale = 1.
    cl_cond = aq_cond / FIX_cond_ratio

    df = pd.DataFrame(index=range(NUM_b),
                      columns=['aq_shape', 'b', 'cond_ratio', 'a_sh',
                               'res. median', 'res. min', 'res. max',
                               'pos. min', 'pos. max'])
    df['aq_shape'] = aq_shape
    df['b'] = b
    df['cond_ratio'] = FIX_cond_ratio

    # fit soft_to_hard parameter
    for i in tqdm(range(NUM_b)):

        # define x-range of the fit
        xi = b[i] / (1 + b[i])
        x_sh = FIX_cond_ratio**(1+1/b[i])
        N = int((1+1/b[i]) * np.log10(FIX_cond_ratio) / DEL_LOG_x)
        x = np.geomspace(x_sh**0.5, x_sh**1.5, N)
        cl_th = B[i]**(1/b[i]) * aq_scale / FIX_cond_ratio * x

        # leat-square optimization
        def fun_to_minimize(a_sh):
            y_ex = q0_exact(cl_cond, cl_th, aq_cond, aq_scale, aq_shape[i],
                            args.aq_para)
            y_ap = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)
            return np.log10(y_ap / aq_cond) - np.log10(y_ex / aq_cond)
        res = least_squares(fun_to_minimize, x0=[1.2])

        # best parameter
        df.loc[i, 'a_sh'] = res.x[0]

        # residual metrics
        df.loc[i, 'res. median'] = 10**np.median(res.fun)
        df.loc[i, 'res. min'] = 10**np.min(res.fun)
        df.loc[i, 'res. max'] = 10**np.max(res.fun)

        # relative position of the min and max errors (0=x_sh^.5, 1=x_sh^1.5)
        imin = np.argmin(res.fun)
        df.loc[i, 'pos. min'] = np.log10(x[imin] / x_sh**0.5) / np.log10(x_sh)
        imax = np.argmax(res.fun)
        df.loc[i, 'pos. max'] = np.log10(x[imax] / x_sh**0.5) / np.log10(x_sh)

    return df


def run(args):

    #fit_ans(args)
    #interpolate_ans()

    df = fit_a_sh(args)
    df.to_csv(args.path / 'a_sh.csv', sep=',', index=True, header=True)
    #interpolate_ash()

def dump_constants(args):
    """Save module constants."""

    d = {            
        'b': [SCA_b, MIN_b, MAX_b, NUM_b],
        'DEL_LOG_x': DEL_LOG_x,
        'FIX_cond_ratio': FIX_cond_ratio
    }   
    json.dump(d, open(args.path / 'constants.txt', 'w'))

def main():

    parser = argparse.ArgumentParser(
                        prog='exp2',
                        description='Fit parameters of the approximate formula')

    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--output', type=str, help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp2_{args.aq_para}'

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
    dump_constants(args)

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()