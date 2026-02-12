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

# Third party imports
import numpy as np
import pandas as pd

# Internal imports
from rate import q_exact_dis
from rate import q0_negl_to_soft
from rate import q0_soft_to_hard
from utils import get_Dless_parameters
q_exact_dis = np.vectorize(q_exact_dis)

####################
# Constants        #
####################

# aquifer shape parameter
SCA_b = 'lin'
MIN_b = 2.1
MAX_b = 11
NUM_b = 15

# dimensionless x ratio
SCA_x = 'log'
MIN_x = -1
MAX_x = 12
NUM_x = 80

# hydraulic conductivity ratio
SCA_cond_ratio = 'log'
MIN_cond_ratio = 0
MAX_cond_ratio = 6
NUM_cond_ratio = 30

####################
# Functions        #
###################

def q0_approximate_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape, C_NSH=None,
                       C_NS_1=-0.3850, C_NS_2=0.2056, C_NS_3=0.5818,
                       C_SH_1=0.4633, C_SH_2=0.5396):

    b = 0.5 * (5 * aq_shape - 1)
    B = (1 - 1 / aq_shape)**2

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_ns = (C_NS_1 + C_NS_2 * b)**C_NS_3
    q0_ns = q0_negl_to_soft(aq_cond, x, xi, a_ns)
    
    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    if C_NSH is not None:
        a_nsh = C_NSH
    else:
        a_nsh = 1.05 + 0.25 * np.tanh(b - 7)
    s = 1 / (1 + (x_sh**0.5 / x)**a_nsh)
    q0 = q0_ns**(1-s) * q0_sh**s

    return q0

def q0_approximate_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                       C_SH_1=0.4633, C_SH_2=0.5396):

    b = 2 + 3 * aq_shape
    B = 1

    xi = b / (1 + b)
    x = B**(-1/b) * cl_th * aq_cond / (aq_scale * cl_cond)
    x_sh = (aq_cond / cl_cond)**(1/xi)

    a_sh = C_SH_1 + C_SH_2 * xi
    q0_sh = q0_soft_to_hard(cl_cond, x, x_sh, xi, a_sh)

    q0 = np.min([aq_cond * np.ones_like(q0_sh), q0_sh], axis=0)

    return q0


def q0_approximate(cl_cond, cl_th, aq_cond, aq_scale, aq_shape, aq_para,
                   C_NSH=None):
    
    if aq_para == 'vGM':
        q0 = q0_approximate_vGM(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                                C_NSH=C_NSH)
    elif aq_para == 'BCB':
        q0 = q0_approximate_BCB(cl_cond, cl_th, aq_cond, aq_scale, aq_shape)

    return q0

def run(args):
    """
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
    df['x'] = x
    df['b'], df['B'], df['xi'], _, df['xsh'] = get_Dless_parameters(
                                    df['cl_cond'], df['cl_th'], df['aq_cond'],
                                    df['aq_scale'], df['aq_shape'],
                                    args.aq_para
                                    )

    # compute rates
    df['q0_ex'] = q_exact_dis(0., cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                              args.aq_para)
    df['q0_ap'] = q0_approximate(cl_cond, cl_th, aq_cond, aq_scale, aq_shape,
                                 args.aq_para, C_NSH=args.C_NSH)
    
    if args.C_NSH is not None:
        name = f'rates_{args.C_NSH}.csv'
    else:
        name = 'rates.csv'
    df.to_csv(args.path / name, sep=',', index=True, header=True)
    print(f'Seepage rates stored in {args.path / name}')

def dump_constants(args):
    """Save module constants which define the regular grid."""

    d = {            
        'b': [SCA_b, MIN_b, MAX_b, NUM_b],
        'x': [SCA_x, MIN_x, MAX_x, NUM_x],
        'cond_ratio': [SCA_cond_ratio, MIN_cond_ratio, MAX_cond_ratio,
                       NUM_cond_ratio]
    }   
    json.dump(d, open(args.path / 'constants.txt', 'w'))

def main():

    parser = argparse.ArgumentParser(
                        prog='exp3',
                        description='Compute seepage rate over the whole' \
                        'infiltrability space')
    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--C_NSH', type=float, default=0., help='Value of' \
                        'sigmoid parameter. Set 0 for using tanh interpolation')
    parser.add_argument('--output', type=str, help='Path to directory')
    parser.add_argument('--clean', default=False, const=True, nargs='?',
                        help='Empty output folder')
    args = parser.parse_args()

    # set default output directory name
    if args.output is None:
        args.output = f'exp3_{args.aq_para}'

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

    # set C_NSH if zero
    if args.C_NSH == 0:
        args.C_NSH = None

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()