#!/usr/bin/python -u

# ./gen_hamm_dct.py
# script generateing NN initialization for training with TNet
#     
# author: Karel Vesely
#

import math, random
import sys


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--dim', dest='dim', help='d1:d2:d3 layer dimensions in the network')
parser.add_option('--gauss', dest='gauss', help='use gaussian noise for weights', action='store_true', default=False)
parser.add_option('--negbias', dest='negbias', help='use uniform [-4.1,-3.9] for bias (defaultall 0.0)', action='store_true', default=False)
parser.add_option('--inputscale', dest='inputscale', help='scale the weights by 3/sqrt(Ninputs)', action='store_true', default=False)
parser.add_option('--linBNdim', dest='linBNdim', help='dim of linear bottleneck (sigmoids will be omitted, bias will be zero)',default=0)
(options, args) = parser.parse_args()

if(options.dim == None):
    parser.print_help()
    sys.exit(1)


dimStrL = options.dim.split(':')

dimL = []
for i in range(len(dimStrL)):
    dimL.append(int(dimStrL[i]))


#print dimL,'linBN',options.linBNdim

for layer in range(len(dimL)-1):
    print '<biasedlinearity>', dimL[layer+1], dimL[layer]
    #weight matrix
    print '['
    for row in range(dimL[layer+1]):
        for col in range(dimL[layer]):
            if(options.gauss):
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
        elif(options.negbias):
            print random.random()/5.0-4.1,
        else:
            print '0.0',
    print ']'

    if(int(options.linBNdim) != dimL[layer+1]):
        if(layer == len(dimL)-2):
            print '<softmax>', dimL[layer+1], dimL[layer+1]
        else:
            print '<sigmoid>', dimL[layer+1], dimL[layer+1]





