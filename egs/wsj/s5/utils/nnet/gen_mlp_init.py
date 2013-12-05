#!/usr/bin/python

# Copyright 2010-2013  Brno University of Technology (author: Karel Vesely)

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

# ./gen_mlp_init.py
# script generateing NN initialization 

import math, random
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--dim', dest='dim', help='d1:d2:d3 layer dimensions in the network')
parser.add_option('--gauss', dest='gauss', help='use gaussian noise for weights', action='store_true', default=False)
parser.add_option('--negbias', dest='negbias', help='use uniform [-4.1,-3.9] for bias (defaultall 0.0)', action='store_true', default=False)
parser.add_option('--inputscale', dest='inputscale', help='scale the weights by 3/sqrt(Ninputs)', action='store_true', default=False)
parser.add_option('--normalized', dest='normalized', help='Generate normalized weights according to X.Glorot paper, U[-x,x] x=sqrt(6)/(sqrt(dim_in+dim_out))', action='store_true', default=False)
parser.add_option('--activation', dest='activation', help='activation type tag (def. <sigmoid>)', default='<sigmoid>')
parser.add_option('--linBNdim', dest='linBNdim', help='dim of linear bottleneck (sigmoids will be omitted, bias will be zero)',default=0)
parser.add_option('--linOutput', dest='linOutput', help='generate MLP with linear output', action='store_true', default=False)
parser.add_option('--seed', dest='seedval', help='seed for random generator',default=0)
(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)

#seeding
seedval=int(options.seedval)
if(seedval != 0):
    random.seed(seedval)


dimStrL = options.dim.split(':')

dimL = []
for i in range(len(dimStrL)):
    dimL.append(int(dimStrL[i]))


#print dimL,'linBN',options.linBNdim
print '<Nnet>'

for layer in range(len(dimL)-1):
    print '<affinetransform>', dimL[layer+1], dimL[layer]
    #precompute...
    nomalized_interval = math.sqrt(6.0) / math.sqrt(dimL[layer+1]+dimL[layer])
    #weight matrix
    print '['
    for row in range(dimL[layer+1]):
        for col in range(dimL[layer]):
            if(options.normalized):
                print random.random()*2.0*nomalized_interval - nomalized_interval, 
            elif(options.gauss):
                if(options.inputscale):
                    print 3/math.sqrt(dimL[layer])*random.gauss(0.0,1.0),
                else:
                    print 0.1*random.gauss(0.0,1.0),
            else:
                if(options.inputscale):
                    print (random.random()-0.5)*2*3/math.sqrt(dimL[layer]),
                else:
                    print random.random()/5.0-0.1, 
        print #newline for each row 
    print ']'
    #bias vector
    print '[',
    for idx in range(dimL[layer+1]):
        if(int(options.linBNdim) == dimL[layer+1]):
            print '0.0',
        elif(layer == len(dimL)-2):#last layer (softmax)
            print '0.0',
        elif(options.negbias):
            print random.random()/5.0-4.1,
        else:
            print '0.0',
    print ']'

    if(int(options.linBNdim) != dimL[layer+1]):
        if(layer == len(dimL)-2):
            if(not(options.linOutput)) :
                print '<softmax>', dimL[layer+1], dimL[layer+1]
        else:
            #print '<sigmoid>', dimL[layer+1], dimL[layer+1]
            print options.activation, dimL[layer+1], dimL[layer+1]

print '</Nnet>'





