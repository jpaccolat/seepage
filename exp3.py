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
from rate import q0_exact
from rate import q0_approximate
from rate import q_modflow
from utils import get_Dless_parameters
from utils import timer

####################
# Constants        #
####################

# aquifer shape parameter
SCA_b = 'lin'
MIN_b = 2.1
MAX_b = 10
NUM_b = 7

# dimensionless x ratio
SCA_x = 'log'
MIN_x = -1
MAX_x = 12
NUM_x = 50

# hydraulic conductivity ratio
SCA_cond_ratio = 'log'
MIN_cond_ratio = 0.5
MAX_cond_ratio = 6
NUM_cond_ratio = 6

####################
# Functions        #
###################

q0_exact = np.vectorize(timer(q0_exact))
q0_approximate = np.vectorize(timer(q0_approximate))
q_modflow = np.vectorize(timer(q_modflow))

def compute_rates(args):
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
    df['b'], df['B'], df['xi'], df['x'], df['xsh'] = get_Dless_parameters(
                                    df['cl_cond'], df['cl_th'], df['aq_cond'],
                                    df['aq_scale'], df['aq_shape'],
                                    args.aq_para
                                    )

    # compute rates
    df['q0_ex'], df['dt_ex'] = q0_exact(cl_cond, cl_th, aq_cond, aq_scale,
                                        aq_shape, args.aq_para)
    df['q0_ap'], df['dt_ap'] = q0_approximate(cl_cond, cl_th, aq_cond, aq_scale,
                                              aq_shape, args.aq_para, args.C_NSH)
    df['q0_mf'], df['dt_mf'] = q_modflow(0., cl_cond, cl_th, aq_cond, aq_scale,
                                         aq_shape, args.aq_para)
    
    df.to_csv(args.path / 'rates.csv', sep=',', index=True, header=True)
    print(f'Seepage rates stored in {args.path / 'rates.csv'}')

def run(args):
    """Run experiment 3."""

    compute_rates(args)

def dump_constants(args):
    """Save module constants which define the regular grid."""

    d = {            
        'b': [SCA_b, MIN_b, MAX_b, NUM_b],
        'x': [SCA_x, MIN_x, MAX_x, NUM_x],
        'cond_ratio': [SCA_cond_ratio, MIN_cond_ratio, MAX_cond_ratio,
                       NUM_cond_ratio],
        'C_NSH': args.C_NSH
    }   
    json.dump(d, open(args.path / 'constants.txt', 'w'))

def main():

    parser = argparse.ArgumentParser(
                        prog='exp3',
                        description='Compute seepage rate over the whole' \
                        'infiltrability space')

    parser.add_argument('--aq_para', type=str, required=True, help='vGM or BCB')
    parser.add_argument('--C_NSH', type=float, default=3.,
                        help='Ponderation exponent for vGM merging.')
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

    # run experiment
    run(args)

    
if __name__ == '__main__':
    main()