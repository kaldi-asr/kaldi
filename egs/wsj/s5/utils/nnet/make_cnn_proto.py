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

import math, random, sys
from optparse import OptionParser

###
### Parse options
###
usage="%prog [options] <feat-dim> <num-leaves> <num-hidden-layers> <num-hidden-neurons>  >nnet-proto-file"
parser = OptionParser(usage)

parser.add_option('--no-softmax', dest='with_softmax', 
                   help='Do not put <SoftMax> in the prototype [default: %default]', 
                   default=True, action='store_false');
parser.add_option('--activation-type', dest='activation_type', 
                   help='Select type of activation function : (<Sigmoid>|<Tanh>) [default: %default]', 
                   default='<Sigmoid>', type='string');
parser.add_option('--hid-bias-mean', dest='hid_bias_mean', 
                   help='Set bias for hidden activations [default: %default]', 
                   default=-2.0, type='float');
parser.add_option('--hid-bias-range', dest='hid_bias_range', 
                   help='Set bias range for hidden activations (+/- 1/2 range around mean) [default: %default]', 
                   default=4.0, type='float');
parser.add_option('--param-stddev-factor', dest='param_stddev_factor', 
                   help='Factor to rescale Normal distriburtion for initalizing weight matrices [default: %default]', 
                   default=0.1, type='float');
parser.add_option('--bottleneck-dim', dest='bottleneck_dim', 
                   help='Make bottleneck network with desired bn-dim (0 = no bottleneck) [default: %default]',
                   default=0, type='int');
parser.add_option('--no-glorot-scaled-stddev', dest='with_glorot', help='Generate normalized weights according to X.Glorot paper, but mapping U->N with same variance (factor sqrt(x/(dim_in+dim_out)))', action='store_false', default=True)
parser.add_option('--no-smaller-input-weights', dest='smaller_input_weights', 
                   help='Disable 1/12 reduction of stddef in input layer [default: %default]',
                   action='store_false', default=True);
parser.add_option('--num-filters1', dest='num_filters1',
		   help='Number of filters in first convolutional layer [default: %default]',
		   default=128, type='int')
parser.add_option('--num-filters2', dest='num_filters2',
		   help='Number of filters in second convolutional layer [default: %default]',
		   default=256, type='int')
parser.add_option('--pool-size', dest='pool_size',
	  	   help='Size of pooling [default: %default]',
		   default=3, type='int')
parser.add_option('--pool-step', dest='pool_step',
		  help='Step of pooling [default: %default]',
		  default=3, type='int')
parser.add_option('--pool-type', dest='pool_type',
		  help='Type of pooling (Max || Average) [default: %default]',
		  default='Max', type='string')
parser.add_option('--pitch-dim', dest='pitch_dim',
		  help='Number of features representing pitch [default: %default]',
		  default=0, type='int')
parser.add_option('--delta-order', dest='delta_order',
		  help='Order of delta features [default: %default]',
		  default=2, type='int')
parser.add_option('--splice', dest='splice',
		  help='Length of splice [default: %default]',
		  default=5,type='int')
parser.add_option('--patch-step1', dest='patch_step1',
		  help='Patch step of first convolutional layer [default: %default]',
		  default=1, type='int')
parser.add_option('--patch-dim1', dest='patch_dim1',
		  help='Lenght of patch of first convolutional layer [default: %default]',
  		  default=9, type='int')
parser.add_option('--dir', dest='dirct',
		  help='Directory, where network prototypes will be saved [default: %default]',
		  default='.', type='string')
parser.add_option('--num-pitch-neurons', dest='num_pitch_neurons',
		  help='Number of neurons in layers processing pitch features [default: %default]',
		  default='200', type='int')



(o,args) = parser.parse_args()
if len(args) != 4 : 
  parser.print_help()
  sys.exit(1)
  
(feat_dim, num_leaves, num_hid_layers, num_hid_neurons) = map(int,args);
### End parse options 

feat_raw_dim = feat_dim / (o.delta_order+1) / (o.splice*2+1) - o.pitch_dim # we need number of feats without deltas and splice and pitch

# Check
assert(feat_dim > 0)
assert(num_leaves > 0)
assert(num_hid_layers >= 0)
assert(num_hid_neurons > 0)
assert(o.pool_type == 'Max' or o.pool_type == 'Average')

# Optionaly scale
def Glorot(dim1, dim2):
  if o.with_glorot:
    # 35.0 = magic number, gives ~1.0 in inner layers for hid-dim 1024dim,
    return 35.0 * math.sqrt(2.0/(dim1+dim2)); 
  else:
    return 1.0



###
### Print prototype of the network
###

# Begin the prototype
print "<NnetProto>"

# Convolutional part of network
num_patch1 = 1 + (feat_raw_dim - o.patch_dim1) / o.patch_step1
num_pool = 1 + (num_patch1 - o.pool_size) / o.pool_step
patch_dim2 = 4 * o.num_filters1
patch_step2 = o.num_filters1
patch_stride2 = num_pool * o.num_filters1 
num_patch2 = 1 + (num_pool * o.num_filters1 - patch_dim2) / patch_step2

convolution_proto = ''  

convolution_proto += "<ConvolutionalComponent> <InputDim> %d <OutputDim> %d <PatchDim> %d <PatchStep> %d <PatchStride> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
			(feat_raw_dim * (o.delta_order+1) * (o.splice*2+1), o.num_filters1 * num_patch1, o.patch_dim1, o.patch_step1, feat_raw_dim, 0.0, 0.0, 0.01)
convolution_proto += "<%sPoolingComponent> <InputDim> %d <OutputDim> %d <PoolSize> %d <PoolStep> %d <PoolStride> %d\n" % \
			(o.pool_type, o.num_filters1*num_patch1, o.num_filters1*num_pool, o.pool_size, o.pool_step, o.num_filters1)
convolution_proto += "<Rescale> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
			(o.num_filters1*num_pool, o.num_filters1*num_pool, 1.0)
convolution_proto += "<AddShift> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
			(o.num_filters1*num_pool, o.num_filters1*num_pool, 0.0)
convolution_proto += "%s <InputDim> %d <OutputDim> %d\n" % \
			(o.activation_type, o.num_filters1*num_pool, o.num_filters1*num_pool)
convolution_proto += "<ConvolutionalComponent> <InputDim> %d <OutputDim> %d <PatchDim> %d <PatchStep> %d <PatchStride> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
			(o.num_filters1*num_pool, o.num_filters2*num_patch2, patch_dim2, patch_step2, patch_stride2, -2.0, 4.0, 0.1)
convolution_proto += "<Rescale> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
			(o.num_filters2 * num_patch2, o.num_filters2*num_patch2, 1.0)
convolution_proto += "<AddShift> <InputDim> %d <OutputDim> %d <InitParam> %f\n" % \
			(o.num_filters2*num_patch2, o.num_filters2*num_patch2, 0.0)
convolution_proto += "%s <InputDim> %d <OutputDim> %d\n" % \
			(o.activation_type, o.num_filters2*num_patch2, o.num_filters2*num_patch2)

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
	((feat_raw_dim + o.pitch_dim) * (o.delta_order+1) * (o.splice*2+1), o.num_pitch_neurons + o.num_filters2*num_patch2, '%s/nnet.proto.convolution' % o.dirct, '%s/nnet.proto.pitch' % o.dirct)

  num_convolution_output = o.num_pitch_neurons + o.num_filters2*num_patch2
else: # no pitch
  print convolution_proto
  
  num_convolution_output = o.num_filters2*num_patch2



# Only last layer (logistic regression)
if num_hid_layers == 0:
  print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
        (num_convolution_output, num_leaves, 0.0, 0.0, (o.param_stddev_factor * Glorot(feat_dim, num_leaves)))
  if o.with_softmax:
    print "<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves)
  print "</NnetProto>"
  # We are done!
  sys.exit(0)

# Assuming we have >0 hidden layers
assert(num_hid_layers > 0)

# First AffineTranform
print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
      (num_convolution_output, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
       (o.param_stddev_factor * Glorot(feat_dim, num_hid_neurons) * \
        (math.sqrt(1.0/12.0) if o.smaller_input_weights else 1.0))) 
      # stddev(U[0,1]) = sqrt(1/12); reducing stddev of weights, 
      # the dynamic range of input data is larger than of a Sigmoid.
print "%s <InputDim> %d <OutputDim> %d" % (o.activation_type, num_hid_neurons, num_hid_neurons)

# Internal AffineTransforms
for i in range(num_hid_layers-1):
  print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
        (num_hid_neurons, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
         (o.param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons)))
  print "%s <InputDim> %d <OutputDim> %d" % (o.activation_type, num_hid_neurons, num_hid_neurons)

# Optionaly add bottleneck
if o.bottleneck_dim != 0:
  assert(o.bottleneck_dim > 0)
  # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
  print "<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f" % \
   (num_hid_neurons, o.bottleneck_dim, \
    (o.param_stddev_factor * Glorot(num_hid_neurons, o.bottleneck_dim) * 0.75 ), 0.1)
  # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
  print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f" % \
   (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
    (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1) 
  print "%s <InputDim> %d <OutputDim> %d" % (o.activation_type, num_hid_neurons, num_hid_neurons)

# Last AffineTransform (10x smaller learning rate on bias)
print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f" % \
      (num_hid_neurons, num_leaves, 0.0, 0.0, \
       (o.param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1)

# Optionaly append softmax
if o.with_softmax:
  print "<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves)

# End the prototype
print "</NnetProto>"

# We are done!
sys.exit(0)

