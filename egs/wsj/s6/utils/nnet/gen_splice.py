#!/usr/bin/python -u

# ./gen_splice.py
# script generates splice nnet transfrom
# 
#     
# author: Karel Vesely
#

from math import *
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--fea-dim', dest='dim_in', help='feature dimension')
parser.add_option('--splice', dest='splice', help='number of frames to concatenate with the central frame')
parser.add_option('--splice-step', dest='splice_step', help='splicing step (frames dont need to be consecutive, --splice 3 --splice-step 2 will select offsets: -6 -4 -2 0 2 4 6)', default='1' )
(options, args) = parser.parse_args()

if(options.dim_in == None):
    parser.print_help()
    sys.exit(1)

dim_in=int(options.dim_in)
splice=int(options.splice)
splice_step=int(options.splice_step)

dim_out=(2*splice+1)*dim_in

print '<splice>', dim_out, dim_in
print '[',

splice_vec = range(-splice*splice_step, splice*splice_step+1, splice_step)
for idx in range(len(splice_vec)):
    print splice_vec[idx],

print ']'

