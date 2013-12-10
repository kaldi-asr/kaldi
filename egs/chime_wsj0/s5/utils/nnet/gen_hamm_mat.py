#!/usr/bin/python -u

# ./gen_hamm_mat.py
# script generates diagonal matrix with hamming window values
#     
# author: Karel Vesely
#

from math import *
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--fea-dim', dest='dim', help='feature dimension')
parser.add_option('--splice', dest='splice', help='applied splice value')
(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)

dim=int(options.dim)
splice=int(options.splice)


#generate the diagonal matrix with hammings
M_2PI = 6.283185307179586476925286766559005

dim_mat=(2*splice+1)*dim
timeContext=2*splice+1
print '['
for row in range(dim_mat):
    for col in range(dim_mat):
        if col!=row:
            print '0',
        else:
            i=int(row/dim)
            print str(0.54 - 0.46*cos((M_2PI * i) / (timeContext-1))),
    print

print ']'


