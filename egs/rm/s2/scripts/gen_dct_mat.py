#!/usr/bin/python -u

# ./gen_dct_mat.py
# script generates matrix with DCT transform
#     
# author: Karel Vesely
#

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


#generate small DCT matrix
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

