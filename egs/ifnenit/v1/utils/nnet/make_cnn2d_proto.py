#!/usr/bin/python

# Copyright 2014  Brno University of Technology (author: Karel Vesely)

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

# Generated Nnet prototype, to be initialized by 'nnet-initialize'.

import math, random, sys, warnings
from optparse import OptionParser

###
### Parse options
###
usage="%prog [options] <feat-dim> <num-leaves> <num-hidden-layers> <num-hidden-neurons>  >nnet-proto-file"
parser = OptionParser(usage)

parser.add_option('--activation-type', dest='activation_type', 
                   help='Select type of activation function : (<Sigmoid>|<Tanh>) [default: %default]', 
                   default='<Sigmoid>', type='string');

parser.add_option('--cnn1-num-filters', dest='cnn1_num_filters',
		   help='Number of filters in first convolutional layer [default: %default]',
		   default=128, type='int')
# this is given by splice
# parser.add_option('--cnn1-fmap-x-len', dest='cnn1_fmap_x_len',
# 	  	   help='Size of cnn1-fmap-x-len [default: %default]',
# 		   default=11, type='int')

# this should be equal to feat_raw_dim
# parser.add_option('--cnn1-fmap-y-len', dest='cnn1_fmap_y_len',
# 	  	   help='Size of cnn1-fmap-y-len [default: %default]',
# 		   default=32, type='int')

parser.add_option('--cnn1-filt-x-len', dest='cnn1_filt_x_len',
	  	   help='Size of cnn1-filt-x-len [default: %default]',
		   default=9, type='int')
parser.add_option('--cnn1-filt-y-len', dest='cnn1_filt_y_len',
	  	   help='Size of cnn1-filt-y-len [default: %default]',
		   default=9, type='int')

parser.add_option('--cnn1-filt-x-step', dest='cnn1_filt_x_step',
	  	   help='Size of cnn1-filt-x-step [default: %default]',
		   default=1, type='int')
parser.add_option('--cnn1-filt-y-step', dest='cnn1_filt_y_step',
	  	   help='Size of cnn1-filt-y-step [default: %default]',
		   default=1, type='int')
parser.add_option('--cnn1-connect-fmap', dest='cnn1_connect_fmap',
	  	   help='Size of cnn1-connect-fmap [default: %default]',
		   default=0, type='int')

parser.add_option('--pool1-x-len', dest='pool1_x_len',
	  	   help='Size of pool1-filt-x-len [default: %default]',
		   default=1, type='int')
parser.add_option('--pool1-x-step', dest='pool1_x_step',
	  	   help='Size of pool1-x-step [default: %default]',
		   default=1, type='int')


# 
parser.add_option('--pool1-y-len', dest='pool1_y_len',
	  	   help='Size of pool1-y-len [default: %default]',
		   default=3, type='int')
parser.add_option('--pool1-y-step', dest='pool1_y_step',
	  	   help='Size of pool1-y-step [default: %default]',
		   default=3, type='int')

parser.add_option('--pool1-type', dest='pool1_type',
		  help='Type of pooling (Max || Average) [default: %default]',
		  default='Max', type='string')

parser.add_option('--cnn2-num-filters', dest='cnn2_num_filters',
		   help='Number of filters in first convolutional layer [default: %default]',
		   default=256, type='int')
parser.add_option('--cnn2-filt-x-len', dest='cnn2_filt_x_len',
	  	   help='Size of cnn2-filt-x-len [default: %default]',
		   default=3, type='int')
parser.add_option('--cnn2-filt-y-len', dest='cnn2_filt_y_len',
	  	   help='Size of cnn2-filt-y-len [default: %default]',
		   default=4, type='int')
parser.add_option('--cnn2-filt-x-step', dest='cnn2_filt_x_step',
	  	   help='Size of cnn2-filt-x-step [default: %default]',
		   default=1, type='int')
parser.add_option('--cnn2-filt-y-step', dest='cnn2_filt_y_step',
	  	   help='Size of cnn2-filt-y-step [default: %default]',
		   default=1, type='int')
parser.add_option('--cnn2-connect-fmap', dest='cnn2_connect_fmap',
	  	   help='Size of cnn2-connect-fmap [default: %default]',
		   default=1, type='int')

parser.add_option('--pitch-dim', dest='pitch_dim',
		  help='Number of features representing pitch [default: %default]',
		  default=0, type='int')
parser.add_option('--delta-order', dest='delta_order',
		  help='Order of delta features [default: %default]',
		  default=2, type='int')
parser.add_option('--splice', dest='splice',
		  help='Length of splice [default: %default]',
		  default=5,type='int')
parser.add_option('--dir', dest='dirct',
		  help='Directory, where network prototypes will be saved [default: %default]',
		  default='.', type='string')
parser.add_option('--num-pitch-neurons', dest='num_pitch_neurons',
		  help='Number of neurons in layers processing pitch features [default: %default]',
		  default='200', type='int')


(o,args) = parser.parse_args()
if len(args) != 1 : 
  parser.print_help()
  sys.exit(1)
  
feat_dim=int(args[0])
### End parse options 

feat_raw_dim = feat_dim / (o.delta_order+1) / (o.splice*2+1) - o.pitch_dim # we need number of feats without deltas and splice and pitch
o.cnn1_fmap_y_len = feat_raw_dim
o.cnn1_fmap_x_len = o.splice*2+1

# Check
assert(feat_dim > 0)
assert(o.pool1_type == 'Max' or o.pool1_type == 'Average')

## Extra checks if dimensions are matching, if not match them by 
## producing a warning
# cnn1
assert( (o.cnn1_fmap_y_len - o.cnn1_filt_y_len) % o.cnn1_filt_y_step == 0 )
assert( (o.cnn1_fmap_x_len - o.cnn1_filt_x_len) % o.cnn1_filt_x_step == 0 )

# subsample1
cnn1_out_fmap_y_len=((1 + (o.cnn1_fmap_y_len - o.cnn1_filt_y_len) / o.cnn1_filt_y_step))
cnn1_out_fmap_x_len=((1 + (o.cnn1_fmap_x_len - o.cnn1_filt_x_len) / o.cnn1_filt_x_step))

# fix filt_len and filt_step
def fix_filt_step(inp_len, filt_len, filt_step):
  
  if ((inp_len - filt_len) % filt_step == 0):
    return filt_step
  else:
    # filt_step <= filt_len
    for filt_step in xrange(filt_len, 0, -1):
      if ((inp_len - filt_len) % filt_step == 0):
        return filt_step
    
o.pool1_y_step = fix_filt_step(cnn1_out_fmap_y_len, o.pool1_y_len, o.pool1_y_step)
if o.pool1_y_step == 1 and o.pool1_y_len != 1:
  warnings.warn('WARNING: Choose different pool1_y_len as subsampling is not happening');
  
o.pool1_x_step = fix_filt_step(cnn1_out_fmap_x_len, o.pool1_x_len, o.pool1_x_step)
if o.pool1_x_step == 1 and o.pool1_x_len != 1:
  warnings.warn('WARNING: Choose different pool1_x_len as subsampling is not happening');


###
### Print prototype of the network
###

# Begin the prototype
print "<NnetProto>"

# Convolutional part of network
'''1st CNN layer'''
cnn1_input_dim=feat_raw_dim * (o.delta_order+1) * (o.splice*2+1)
cnn1_out_fmap_x_len=((1 + (o.cnn1_fmap_x_len - o.cnn1_filt_x_len) / o.cnn1_filt_x_step))
cnn1_out_fmap_y_len=((1 + (o.cnn1_fmap_y_len - o.cnn1_filt_y_len) / o.cnn1_filt_y_step))
cnn1_output_dim=o.cnn1_num_filters * cnn1_out_fmap_x_len * cnn1_out_fmap_y_len

'''1st Pooling layer'''
pool1_input_dim=cnn1_output_dim
pool1_fmap_x_len=cnn1_out_fmap_x_len
pool1_out_fmap_x_len=((1 + (pool1_fmap_x_len - o.pool1_x_len) / o.pool1_x_step))
pool1_fmap_y_len=cnn1_out_fmap_y_len
pool1_out_fmap_y_len=((1 + (pool1_fmap_y_len - o.pool1_y_len) / o.pool1_y_step))
pool1_output_dim=o.cnn1_num_filters*pool1_out_fmap_x_len*pool1_out_fmap_y_len

'''2nd CNN layer'''
cnn2_input_dim=pool1_output_dim
cnn2_fmap_x_len=pool1_out_fmap_x_len
cnn2_out_fmap_x_len=((1 + (cnn2_fmap_x_len - o.cnn2_filt_x_len) / o.cnn2_filt_x_step))
cnn2_fmap_y_len=pool1_out_fmap_y_len
cnn2_out_fmap_y_len=((1 + (cnn2_fmap_y_len - o.cnn2_filt_y_len) / o.cnn2_filt_y_step))
cnn2_output_dim=o.cnn2_num_filters * cnn2_out_fmap_x_len * cnn2_out_fmap_y_len


convolution_proto = ''

convolution_proto += "<Convolutional2DComponent> <InputDim> %d <OutputDim> %d <FmapXLen> %d <FmapYLen> %d <FiltXLen> %d <FiltYLen> %d <FiltXStep> %d <FiltYStep> %d <ConnectFmap> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
    ( cnn1_input_dim, cnn1_output_dim, o.cnn1_fmap_x_len, o.cnn1_fmap_y_len, o.cnn1_filt_x_len, o.cnn1_filt_y_len, o.cnn1_filt_x_step, o.cnn1_filt_y_step, o.cnn1_connect_fmap, 0.0, 0.0, 0.01 )
convolution_proto += "<%sPooling2DComponent> <InputDim> %d <OutputDim> %d <FmapXLen> %d <FmapYLen> %d <PoolXLen> %d <PoolYLen> %d <PoolXStep> %d <PoolYStep> %d\n" % \
    ( o.pool1_type, pool1_input_dim, pool1_output_dim, pool1_fmap_x_len, pool1_fmap_y_len, o.pool1_x_len, o.pool1_y_len, o.pool1_x_step, o.pool1_y_step )
convolution_proto += "<Rescale> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
    ( pool1_output_dim, pool1_output_dim, 1.0 )
convolution_proto += "<AddShift> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
    ( pool1_output_dim, pool1_output_dim, 0.0 )
convolution_proto += "%s <InputDim> %d <OutputDim> %d\n" % \
    ( o.activation_type, pool1_output_dim, pool1_output_dim )
convolution_proto += "<Convolutional2DComponent> <InputDim> %d <OutputDim> %d <FmapXLen> %d <FmapYLen> %d <FiltXLen> %d <FiltYLen> %d <FiltXStep> %d <FiltYStep> %d <ConnectFmap> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
    ( cnn2_input_dim, cnn2_output_dim, cnn2_fmap_x_len, cnn2_fmap_y_len, o.cnn2_filt_x_len, o.cnn2_filt_y_len, o.cnn2_filt_x_step, o.cnn2_filt_y_step, o.cnn2_connect_fmap, -2.0, 4.0, 0.1 )
convolution_proto += "<Rescale> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
    ( cnn2_output_dim, cnn2_output_dim, 1.0)
convolution_proto += "<AddShift> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
    ( cnn2_output_dim, cnn2_output_dim, 0.0)
convolution_proto += "%s <InputDim> %d <OutputDim> %d\n" % \
    ( o.activation_type, cnn2_output_dim, cnn2_output_dim)

if (o.pitch_dim > 0):
  # convolutional part
  f_conv = open('%s/nnet.proto.convolution' % o.dirct, 'w')
  f_conv.write('<NnetProto>\n')
  f_conv.write(convolution_proto)
  f_conv.write('</NnetProto>\n')
  f_conv.close()
  
  # pitch part
  f_pitch = open('%s/nnet.proto.pitch' % o.dirct, 'w')
  f_pitch.write('<NnetProto>\n')
  f_pitch.write('<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n' % \
		((o.pitch_dim * (o.delta_order+1) * (o.splice*2+1)), o.num_pitch_neurons, -2.0, 4.0, 0.109375))
  f_pitch.write('%s <InputDim> %d <OutputDim> %d\n' % \
		(o.activation_type, o.num_pitch_neurons, o.num_pitch_neurons))
  f_pitch.write('<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n' % \
		(o.num_pitch_neurons, o.num_pitch_neurons, -2.0, 4.0, 0.109375))
  f_pitch.write('%s <InputDim> %d <OutputDim> %d\n' % \
		(o.activation_type, o.num_pitch_neurons, o.num_pitch_neurons))
  f_pitch.write('</NnetProto>\n')
  f_pitch.close()

  # paralell part
  vector = ''
  for i in range(1, (feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), feat_raw_dim + o.pitch_dim):
    vector += '%d:1:%d ' % (i, i + feat_raw_dim - 1)
  for i in range(feat_raw_dim+1, (feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), feat_raw_dim + o.pitch_dim):
    vector += '%d:1:%d ' % (i, i + o.pitch_dim - 1)
  print '<Copy> <InputDim> %d <OutputDim> %d <BuildVector>  %s </BuildVector> ' % \
	((feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), (feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), vector)
  print '<ParallelComponent> <InputDim> %d <OutputDim> %d <NestedNnetProto> %s %s </NestedNnetProto>' % \
	((feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), o.num_pitch_neurons + cnn2_output_dim, '%s/nnet.proto.convolution' % o.dirct, '%s/nnet.proto.pitch' % o.dirct)

  num_convolution_output = o.num_pitch_neurons + cnn2_output_dim
else: # no pitch
  print convolution_proto

# We are done!
sys.exit(0)


