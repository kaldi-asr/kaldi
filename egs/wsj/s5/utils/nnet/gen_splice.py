#!/usr/bin/env python

# Copyright 2012  Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# ./gen_splice.py
# generates <splice> Component

from __future__ import print_function
from math import *
import sys


from optparse import OptionParser

def print_on_same_line(text):
    if (sys.version_info > (3,0)):
        print(text, end=' ')
    else:
        print text,

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

print('<splice> {0} {1}'.format(dim_out, dim_in))
print_on_same_line('[')

splice_vec = list(range(-splice*splice_step, splice*splice_step+1, splice_step))
for idx in range(len(splice_vec)):
    print_on_same_line(splice_vec[idx])

print(']')

