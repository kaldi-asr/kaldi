#!/usr/bin/env python3

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# License: Apache 2.0.

import os
import argparse
import subprocess
import sys
import re


parser = argparse.ArgumentParser(description="This script works out the embedding dimension from a "
                                 "nnet3 neural network (e.g. 0.raw).  It does this by invoking "
                                 "nnet3-info to print information about the neural network, and "
                                 "parsing it.  You should make sure nnet3-info is on your path "
                                 "before you call this script.  It is an error if the input and "
                                 "output dimensions of the neural network are not the same.  This "
                                 "script prints the embedding dimension to the standard output.",
                                 epilog="E.g. " + sys.argv[0] + " 0.raw",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("nnet",
                    help="Path for raw neural net (e.g. 0.raw)")

args = parser.parse_args()

if not os.path.exists(args.nnet):
    sys.exit(sys.argv[0] + ": input neural net '{0}' does not exist.".format(args.nnet))

proc = subprocess.Popen(["nnet3-info", args.nnet], stdout=subprocess.PIPE)
out_lines = proc.stdout.readlines()
proc.communicate()
if proc.returncode != 0:
    sys.exit(sys.argv[0] + ": error running command 'nnet3-info {0}'".format(args.nnet))


# we're looking for lines like:
# input-node name=input dim=600
# output-node name=output input=output.affine dim=600

input_dim=-1
output_dim=-1
left_context=0
right_context=0
for line in out_lines:
    line = line.decode('utf-8')
    m = re.search(r'input-node name=input dim=(\d+)', line)
    if m is not None:
        try:
            input_dim = int(m.group(1))
        except:
            sys.exit(sys.argv[0] + ": error processing line {0}".format(line))

    m = re.search(r'output-node name=output .* dim=(\d+)', line)
    if m is not None:
        try:
            output_dim = int(m.group(1))
        except:
            sys.exit(sys.argv[0] + ": error processing line {0}".format(line))

    m = re.match(r'left-context: (\d+)', line)
    if m is not None:
        left_context = int(m.group(1))
    m = re.match(r'right-context: (\d+)', line)
    if m is not None:
        right_context = int(m.group(1))

if input_dim == -1:
    sys.exit(sys.argv[0] + ": could not get input dim from output "
             "of 'nnet3-info {0}'".format(args.nnet))

if output_dim == -1:
    sys.exit(sys.argv[0] + ": could not get output dim from output "
             "of 'nnet3-info {0}'".format(args.nnet))

if left_context == -1:
    sys.exit(sys.argv[0] + ": could not get left context output "
             "of 'nnet3-info {0}'".format(args.nnet))

if right_context == -1:
    sys.exit(sys.argv[0] + ": could not get right context output "
             "of 'nnet3-info {0}'".format(args.nnet))

if right_context > 0:
    sys.exit(sys.argv[0] + ": right-context of model {0} is >0: (it's {1}). "
             "This model cannot be used as an RNNLM because it sees the "
             "future.".format(args.nnet, left_context))

if left_context > 0:
    sys.exit(sys.argv[0] + ": left-context of model {0} is >0: (it's {1}). "
             "This model cannot be used as an RNNLM because it requires left "
             "context and the code does not support this.  You can generally use "
             "IfDefined() in descriptors, and set configs of layers, in such "
             "a way that left-context is not required"
             "".format(args.nnet, left_context))

if input_dim != output_dim:
    sys.exit(sys.argv[0] + ": input and output dims differ for "
             "nnet '{0}': {1} != {2}".format(
            args.nnet, input_dim, output_dim))

print('{}'.format(input_dim))
