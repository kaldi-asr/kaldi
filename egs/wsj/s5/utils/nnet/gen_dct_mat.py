#!/usr/bin/python

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

# ./gen_dct_mat.py
# script generates matrix with DCT transform, which is sparse 
# and takes into account that data-layout is along frequency axis, 
# while DCT is done along temporal axis.

from math import *
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--fea-dim', dest='dim', help='feature dimension')
parser.add_option('--splice', dest='splice', help='applied splice value')
parser.add_option('--dct-basis', dest='dct_basis', help='number of DCT basis')
(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)

dim=int(options.dim)
splice=int(options.splice)
dct_basis=int(options.dct_basis)

timeContext=2*splice+1


#generate the DCT matrix
M_PI = 3.1415926535897932384626433832795
M_SQRT2 = 1.4142135623730950488016887


#generate sparse DCT matrix
print '['
for k in range(dct_basis):
    for m in range(dim):
        for n in range(timeContext):
          if(n==0): 
              print m*'0 ',
          else: 
              print (dim-1)*'0 ',
          print str(sqrt(2.0/timeContext)*cos(M_PI/timeContext*k*(n+0.5))),
          if(n==timeContext-1):
              print (dim-m-1)*'0 ',
        print
    print 

print ']'

