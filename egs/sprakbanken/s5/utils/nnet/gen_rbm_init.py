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

#
# Initialization of single RBM (``layer'')
#

import math, random
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--dim', dest='dim', help='d1:d2 layer dimensions in the network')
parser.add_option('--gauss', dest='gauss', help='use gaussian noise for weights', action='store_true', default=False)
parser.add_option('--gauss-scale', dest='gauss_scale', help='standard deviation of the gaussain noise', default='0.1')
parser.add_option('--negbias', dest='negbias', help='use uniform [-4.1,-3.9] for bias (default all 0.0)', action='store_true', default=False)
parser.add_option('--hidtype', dest='hidtype', help='gauss/bern', default='bern')
parser.add_option('--vistype', dest='vistype', help='gauss/bern', default='bern')
parser.add_option('--cmvn-nnet', dest='cmvn_nnet', help='cmvn_nnet to parse mean activation used in visible bias initialization', default='')


(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)


dimStrL = options.dim.split(':')
assert(len(dimStrL) == 2) #only single layer to initialize
dimL = []
for i in range(len(dimStrL)):
    dimL.append(int(dimStrL[i]))

gauss_scale=float(options.gauss_scale)

#generate RBM
print '<rbm>', dimL[1], dimL[0]
print options.vistype, options.hidtype


#init weight matrix
print '['
for row in range(dimL[1]):
    for col in range(dimL[0]):
        if(options.gauss):
            print gauss_scale * random.gauss(0.0,1.0),
        else:
            print (random.random()-0.5)/5.0, 
    print
print ']'

#init visbias
if len(options.cmvn_nnet)>0:
    ### use the formula log(p/(1-p) for visible biases, where p is mean activity of the neuron
    f = open(options.cmvn_nnet)
    #make sure file starts by <addshift>
    line = f.readline()
    arr = line.split(' ')
    if arr[0] == '<Nnet>': #optionally skip <Nnet>
        line = f.readline()
        arr = line.split(' ')
    if arr[0].lower() != '<addshift>':
        raise Exception('missing <addshift> in '+options.cmvn_nnet)
    #get the p's
    line = f.readline()
    arr = line.strip().split(' ')
    assert(len(arr)-2 == dimL[0])
    #print the values
    print '[',
    for i in range(1,len(arr)-1):
        p = -float(arr[i])
        #p must be after sigmoid
        if(not (p >= 0.0 and p <= 1.0)):
            raise Exception('Negative addshifts from '+options.cmvn_nnet+' must be 0..1, ie. the sigmoid outputs')
        #limit the bias to +/- 8, we will modify the p values accordingly
        if(p < 0.00033535):
            p = 0.00033535
        if(p > 0.99966465):
            p = 0.99966465
        #use the inverse sigmoid to get biases from the mean activations
        print math.log(p/(1-p)),
    print ']'
    f.close()
else:
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

