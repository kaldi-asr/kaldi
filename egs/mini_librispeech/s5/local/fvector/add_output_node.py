#!/usr/bin/env/python

from __future__ import print_function
import argparse
import logging
import os
import pprint
import sys
import shutil
import traceback

def get_args():
    parser = argparse.ArgumentParser(description="Add the S and b output node "
            "which is used in plda object function.",
            epilog="Called by local/fvector/run_fvector.sh")
    parser.add_argument("--input-dim", type=int, required=True,
            help="The input dimension of fvector network.")
    parser.add_argument("--output-dim", type=int, required=True,
            help="The output dimension of fvector network which is used to "
            "compute the dimension of S matrix.")
    parser.add_argument("--s-scale", type=float, default=0.2,
            help="Scaling factor on the output 's' (s is a symmetric matrix "
            "used for scoring).")
    parser.add_argument("--b-scale", type=float, default=0.2,
            help="Scaling factor on output 'b' (b is a scalar offset used in scoring).")
    parser.add_argument("--config-file", type=str, required=True,
            help="The file is needed to be modified. It's always is configs/final.config")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    f = open(args.config_file, "a")
    # The s output
    s_dim = (args.output_dim) * (args.output_dim+1) / 2

    print('component name=x-s type=ConstantFunctionComponent input-dim={0} output-dim={1} '
          'output-mean=0 output-stddev=0 '.format(
              args.input_dim, s_dim), file=f)
    print('component-node name=x-s component=x-s input=IfDefined(input)',           
          file=f)                                                                   
    print('component name=x-s-scale type=FixedScaleComponent dim={0} scale={1}'.format(
                s_dim, args.s_scale), file=f);                                      
    print('component-node name=x-s-scale component=x-s-scale input=x-s',            
          file=f)                                                                   
    print('output-node name=s input=x-s-scale', file=f)                             
                                                                                 
    # now the 'b' output, which is just a scalar.                                   
    b_dim = 1                                                                       
    print('component name=x-b type=ConstantFunctionComponent input-dim={0} output-dim=1 '
          'output-mean=0 output-stddev=0 '.format(args.input_dim), file=f)           
    print('component-node name=x-b component=x-b input=IfDefined(input)', file=f)   
    print('component name=x-b-scale type=FixedScaleComponent dim=1 scale={0}'.format(
            args.b_scale), file=f);                                                 
    print('component-node name=x-b-scale component=x-b-scale input=x-b',            
          file=f)                                                                   
    print('output-node name=b input=x-b-scale', file=f)                             
    f.close()                                                      



if __name__ == "__main__":
    main()
