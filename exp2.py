#!/usr/bin/env python

"""Second experiment.

In this module, the parameters of the approximate formula are fitted.

The exponent a_ns from the formula q0_ns is fitted for the vGM parametrization.
The exponent a_sh from the formula q0_sh is fitted for the vGM and the BCB 
parametrizations. The fits are obtained for a range of shape parameters. The
limit of infinite Ka/Kc ratio is considered (practicaly set to 10^12).

Best fits are saved in the files a_ns.csv and a_sh.csv respectively, together
with min and max values of q0_approx / q0_exact. Also are stored the x-values
where the min and max occurs, realtive to the interval boundaries. Fit intervals
for q0_ns and q0_sh are respectively x_sh^-0.5 to x_sh^0.5 and x_sh^0.5 to
x_sh^1.5. These bounds are chosen to equally span the regimes.

Fits of a_ns(b) and a_sh(b) are given in the companion notebook.
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
from rate import q_exact_full
from rate import q0_negl_to_soft
from rate import q0_soft_to_hard
from rate import q0_approx_full_vGM

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

def fit_a_ns(args):
    """
    Fit the parameter a_ns occuring in the approximate forumla q0_ns. 

    The best fit is computed for a range of shape parameters together with min
    and max values of q0_ns / q0_exact. Also are stored the x-values where the
    min and max occurs, realtive to the interval boundaries.

    The fit data is written in a csv file.
    """

    if args.aq_para != 'vGM':
        print(f'a_ns is not defined for {args.aq_para}. Skip computation.')
        return 0

    # interval to scan
    b = np.linspace(MIN_b, MAX_b, NUM_b)

    # convert to shape parameters
    aq_shape = (2 * b + 1) / 5
    B = (1 - 1 / aq_shape)**2

    # set inputs
    aq_cond = 1.
    aq_scale = 1.
    cl_cond = aq_cond / FIX_cond_ratio

    df_ns = pd.DataFrame(index=range(NUM_b))
    df_ns['aq_shape'] = aq_shape
    df_ns['b'] = b
    df_ns['cond_ratio'] = FIX_cond_ratio
    df_sh = df_ns.copy()

    # fit parameters
    for i in tqdm(range(NUM_b)):

        # define x-range of the fit
        xi = b[i] / (1 + b[i])
        x_sh = FIX_cond_ratio**(1+1/b[i])
        N = int((1+1/b[i]) * np.log10(FIX_cond_ratio) / DEL_LOG_x)
        x = np.geomspace(x_sh**-0.5, x_sh**0.5, N)
        cl_th = B[i]**(1/b[i]) * aq_scale / FIX_cond_ratio * x

        # fit a_ns
        def fun_to_minimize(a_ns):
            y_ex = q_exact_full(0., cl_cond, cl_th, aq_cond, aq_scale,
                               aq_shape[i], 'vGM')
            y_ap = q0_negl_to_soft(aq_cond, x, xi, a_ns)
            return np.log10(y_ap / aq_cond) - np.log10(y_ex / aq_cond)
        res = least_squares(fun_to_minimize, x0=[1.], method='lm')

        # best parameter
        df_ns.loc[i, 'a_ns'] = res.x[0]

        # residual metrics
        df_ns.loc[i, 'res. min'] = 10**np.min(res.fun)
        df_ns.loc[i, 'res. max'] = 10**np.max(res.fun)

        # relative position of the min and max errors (0=x_sh^.5, 1=x_sh^1.5)
        imin = np.argmin(res.fun)
        df_ns.loc[i, 'pos. min'] = np.log10(x[imin]*x_sh**0.5) / np.log10(x_sh)
        imax = np.argmax(res.fun)
        df_ns.loc[i, 'pos. max'] = np.log10(x[imax]*x_sh**0.5) / np.log10(x_sh)

    df_ns.to_csv(args.path / 'a_ns.csv', sep=',', index=True, header=True)
    print(f'a_ns values stored in {args.path / 'a_ns.csv'}')

def fit_a_sh(args):
    """
    Fit the parameter a_sh occuring in the approximate forumla q0_sh. 

    The best fit is computed for a range of shape parameters together with min
    and max values of q0_sh / q0_exact. Also are stored the x-values where the
    min and max occurs, realtive to the interval boundaries.

    The fit data is written in a csv file.
    """

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

    df_sh = pd.DataFrame(index=range(NUM_b))
    df_sh['aq_shape'] = aq_shape
    df_sh['b'] = b
    df_sh['cond_ratio'] = FIX_cond_ratio

    # fit parameters
    for i in tqdm(range(NUM_b)):

        # define x-range of the fit
        xi = b[i] / (1 + b[i])
        x_sh = FIX_cond_ratio**(1+1/b[i])
        N = int((1+1/b[i]) * np.log10(FIX_cond_ratio) / DEL_LOG_x)
        x = np.geomspace(x_sh**0.5, x_sh**1.5, N)
        cl_th = B[i]**(1/b[i]) * aq_scale / FIX_cond_ratio * x
     
        # fit a_sh
        def fun_to_minimize(a_sh):
            y_ex = q_exact_full(0., cl_cond, cl_th, aq_cond, aq_scale,
                               aq_shape[i], args.aq_para)
            y_ap = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)
            return np.log10(y_ap / aq_cond) - np.log10(y_ex / aq_cond)
        res = least_squares(fun_to_minimize, x0=[1.2], method='lm')

        # best parameter
        df_sh.loc[i, 'a_sh'] = res.x[0]

        # residual metrics
        df_sh.loc[i, 'res. min'] = 10**np.min(res.fun)
        df_sh.loc[i, 'res. max'] = 10**np.max(res.fun)

        # relative position of the min and max errors (0=x_sh^.5, 1=x_sh^1.5)
        imin = np.argmin(res.fun)
        df_sh.loc[i, 'pos. min'] = np.log10(x[imin]/x_sh**0.5) / np.log10(x_sh)
        imax = np.argmax(res.fun)
        df_sh.loc[i, 'pos. max'] = np.log10(x[imax]/x_sh**0.5) / np.log10(x_sh)

    df_sh.to_csv(args.path / 'a_sh.csv', sep=',', index=True, header=True)
    print(f'a_sh values stored in {args.path / 'a_sh.csv'}')

def fit_a_nsh(args):
    """To be done"""

    if args.aq_para != 'vGM':
        print(f'a_nsh is not defined for {args.aq_para}. Skip computation.')
        return 0
    
    # interval to scan
    b = np.linspace(MIN_b, MAX_b, NUM_b)
    aq_shape = (2 * b + 1) / 5
    B = (1 - 1 / aq_shape)**2
    
    # set inputs
    aq_cond = 1.
    aq_scale = 1.
    cl_cond = aq_cond / FIX_cond_ratio

    df = pd.DataFrame(index=range(NUM_b))
    df['aq_shape'] = aq_shape
    df['b'] = b
    df['cond_ratio'] = FIX_cond_ratio

    # fit parameters
    for i in tqdm(range(NUM_b)):
        # define x-range of the fit
        x_sh = FIX_cond_ratio**(1+1/b[i])
        N = int((1+1/b[i]) * np.log10(FIX_cond_ratio) / DEL_LOG_x)
        x = np.geomspace(1, x_sh, N)
        cl_th = B[i]**(1/b[i]) * aq_scale / FIX_cond_ratio * x

        # fit a_nsh
        def fun_to_minimize(a_nsh):
            y_ex = q_exact_full(0., cl_cond, cl_th, aq_cond, aq_scale,
                               aq_shape[i], 'vGM')
            y_ap = q0_approx_full_vGM(cl_cond, cl_th, aq_cond, aq_scale,
                                      aq_shape[i], a_nsh)
            return np.log10(y_ap / aq_cond) - np.log10(y_ex / aq_cond)
        res = least_squares(fun_to_minimize, x0=[2], method='lm')

        print(b[i])
        print(res.message)

        # best parameter
        df.loc[i, 'a_nsh'] = res.x[0]
        df.loc[i, 'cost'] = res.cost

        # residual metrics
        df.loc[i, 'res. min'] = 10**np.min(res.fun)
        df.loc[i, 'res. max'] = 10**np.max(res.fun)

        # relative position of the min and max errors (0=x_sh^.5, 1=x_sh^1.5)
        imin = np.argmin(res.fun)
        df.loc[i, 'pos. min'] = np.log10(x[imin]) / np.log10(x_sh)
        imax = np.argmax(res.fun)
        df.loc[i, 'pos. max'] = np.log10(x[imax]) / np.log10(x_sh)

    df.to_csv(args.path / 'a_nsh.csv', sep=',', index=True, header=True)
    print(f'a_nsh values stored in {args.path / 'a_nsh.csv'}')

def run(args):
    """Run experiment 2."""

    if 'ns' in args.fit:
        fit_a_ns(args)

    if 'sh' in args.fit:
        fit_a_sh(args)

    if 'nsh' in args.fit:
        fit_a_nsh(args)

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
    parser.add_argument('--fit', type=str, nargs='*',
                        default=['ns', 'sh'],
                        help='List of fits to perform')
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