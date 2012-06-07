#!/usr/bin/python -u

# egs/wsj/s2/scripts/gen_rbm_init.py
#
# Copyright 2012  Karel Vesely
#
# Initializes the RBM Neural Network
#
# calling example:
# python gen_mlp_init.py --dimIn=598 --dimOut=135 --dimHid=1024:1024:1024 
#

import math, random
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--dim', dest='dim', help='d1:d2 layer dimensions in the network')
parser.add_option('--gauss', dest='gauss', help='use gaussian noise for weights', action='store_true', default=False)
parser.add_option('--negbias', dest='negbias', help='use uniform [-4.1,-3.9] for bias (default all 0.0)', action='store_true', default=False)
parser.add_option('--hidtype', dest='hidtype', help='gauss/bern', default='bern')
parser.add_option('--vistype', dest='vistype', help='gauss/bern', default='bern')

(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)


dimStrL = options.dim.split(':')
assert(len(dimStrL) == 2) #only single layer to initialize
dimL = []
for i in range(len(dimStrL)):
    dimL.append(int(dimStrL[i]))


#generate RBM
print '<rbm>', dimL[1], dimL[0]
print options.vistype, options.hidtype


#init weight matrix
print '['
for row in range(dimL[1]):
    for col in range(dimL[0]):
        if(options.gauss):
            print 0.1*random.gauss(0.0,1.0),
        else:
            print random.random()/5.0-0.1, 
    print
print ']'

#init visbias
print '[',
for idx in range(dimL[0]):
    if(options.vistype=='gauss'):
        print '0.0',
    elif(options.negbias):
        print random.random()/5.0-4.1,
    else:
        print '0.0',
print ']'

#init hidbias
print '[',
for idx in range(dimL[1]):
    if(options.hidtype=='gauss'):
        print '0.0',
    elif(options.negbias):
        print random.random()/5.0-4.1,
    else:
        print '0.0',
print ']'

