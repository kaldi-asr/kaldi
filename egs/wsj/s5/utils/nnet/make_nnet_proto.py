#!/usr/bin/env python

# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)

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

from __future__ import division
from __future__ import print_function
import math, random, sys, re

###
### Parse options
###
from optparse import OptionParser
usage="%prog [options] <feat-dim> <num-leaves> <num-hid-layers> <num-hid-neurons> >nnet-proto-file"
parser = OptionParser(usage)

# Softmax related,
parser.add_option('--no-softmax', dest='with_softmax',
                   help='Do not put <SoftMax> in the prototype [default: %default]',
                   default=True, action='store_false');
parser.add_option('--block-softmax-dims', dest='block_softmax_dims',
                   help='Generate <BlockSoftmax> with dims D1:D2:D3 [default: %default]',
                   default="", type='string');
# Activation related,
parser.add_option('--activation-type', dest='activation_type',
                   help='Select type of activation function : (<Sigmoid>|<Tanh>|<ParametricRelu>) [default: %default]',
                   default='<Sigmoid>', type='string');
parser.add_option('--activation-opts', dest='activation_opts',
                   help='Additional options for protoype of activation function [default: %default]',
                   default='', type='string');
# Affine-transform related,
parser.add_option('--hid-bias-mean', dest='hid_bias_mean',
                   help='Set bias for hidden activations [default: %default]',
                   default=-2.0, type='float');
parser.add_option('--hid-bias-range', dest='hid_bias_range',
                   help='Set bias range for hidden activations (+/- 1/2 range around mean) [default: %default]',
                   default=4.0, type='float');
parser.add_option('--param-stddev-factor', dest='param_stddev_factor',
                   help='Factor to rescale Normal distriburtion for initalizing weight matrices [default: %default]',
                   default=0.1, type='float');
parser.add_option('--no-glorot-scaled-stddev', dest='with_glorot',
                   help='Generate normalized weights according to X.Glorot paper, but mapping U->N with same variance (factor sqrt(x/(dim_in+dim_out)))',
                   action='store_false', default=True);
parser.add_option('--no-smaller-input-weights', dest='smaller_input_weights',
                   help='Disable 1/12 reduction of stddef in input layer [default: %default]',
                   action='store_false', default=True);
parser.add_option('--no-bottleneck-trick', dest='bottleneck_trick',
                   help='Disable smaller initial weights and learning rate around bottleneck',
                   action='store_false', default=True);
parser.add_option('--max-norm', dest='max_norm',
                   help='Max radius of neuron-weights in L2 space (if longer weights get shrinked, not applied to last layer, 0.0 = disable) [default: %default]',
                   default=0.0, type='float');
parser.add_option('--affine-opts', dest='affine_opts',
                   help='Additional options for protoype of affine tranform [default: %default]',
                   default='', type='string');
# Topology related,
parser.add_option('--bottleneck-dim', dest='bottleneck_dim',
                   help='Make bottleneck network with desired bn-dim (0 = no bottleneck) [default: %default]',
                   default=0, type='int');
parser.add_option('--with-dropout', dest='with_dropout',
                   help='Add <Dropout> after the non-linearity of hidden layer.',
                   action='store_true', default=False);
parser.add_option('--dropout-opts', dest='dropout_opts',
                   help='Extra options for dropout [default: %default]',
                   default='', type='string');


(o,args) = parser.parse_args()
if len(args) != 4 :
  parser.print_help()
  sys.exit(1)

# A HACK TO PASS MULTI-WORD OPTIONS, WORDS ARE CONNECTED BY UNDERSCORES '_',
o.activation_opts = o.activation_opts.replace("_"," ")
o.affine_opts = o.affine_opts.replace("_"," ")
o.dropout_opts = o.dropout_opts.replace("_"," ")

(feat_dim, num_leaves, num_hid_layers, num_hid_neurons) = [int(i) for i in args];
### End parse options


# Check
assert(feat_dim > 0)
assert(num_leaves > 0)
assert(num_hid_layers >= 0)
assert(num_hid_neurons > 0)
if o.block_softmax_dims:
  assert(sum(map(int, re.split("[,:]", o.block_softmax_dims))) == num_leaves) # posible separators : ',' ':'

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

# NO HIDDEN LAYER, ADDING BOTTLENECK!
# No hidden layer while adding bottleneck means:
# - add bottleneck layer + hidden layer + output layer
if num_hid_layers == 0 and o.bottleneck_dim != 0:
  assert(o.bottleneck_dim > 0)
  assert(num_hid_layers == 0)
  if o.bottleneck_trick:
    # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
    print("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f" % \
     (feat_dim, o.bottleneck_dim, \
      (o.param_stddev_factor * Glorot(feat_dim, o.bottleneck_dim) * 0.75 ), 0.1))
    # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
    print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f <MaxNorm> %f" % \
     (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
      (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1, o.max_norm))
  else:
    print("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f" % \
     (feat_dim, o.bottleneck_dim, \
      (o.param_stddev_factor * Glorot(feat_dim, o.bottleneck_dim))))
    print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f" % \
     (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
      (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons)), o.max_norm))
  print("%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts)) # Non-linearity
  # Last AffineTransform (10x smaller learning rate on bias)
  print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f" % \
   (num_hid_neurons, num_leaves, 0.0, 0.0, \
    (o.param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1))
  # Optionaly append softmax
  if o.with_softmax:
    if o.block_softmax_dims == "":
      print("<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves))
    else:
      print("<BlockSoftmax> <InputDim> %d <OutputDim> %d <BlockDims> %s" % (num_leaves, num_leaves, o.block_softmax_dims))
  print("</NnetProto>")
  # We are done!
  sys.exit(0)

# NO HIDDEN LAYERS!
# Add only last layer (logistic regression)
if num_hid_layers == 0:
  print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
        (feat_dim, num_leaves, 0.0, 0.0, (o.param_stddev_factor * Glorot(feat_dim, num_leaves))))
  if o.with_softmax:
    if o.block_softmax_dims == "":
      print("<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves))
    else:
      print("<BlockSoftmax> <InputDim> %d <OutputDim> %d <BlockDims> %s" % (num_leaves, num_leaves, o.block_softmax_dims))
  print("</NnetProto>")
  # We are done!
  sys.exit(0)


# THE USUAL DNN PROTOTYPE STARTS HERE!
# Assuming we have >0 hidden layers,
assert(num_hid_layers > 0)

# Begin the prototype,
# First AffineTranform,
print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
      (feat_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
       (o.param_stddev_factor * Glorot(feat_dim, num_hid_neurons) * \
        (math.sqrt(1.0/12.0) if o.smaller_input_weights else 1.0)), o.max_norm, o.affine_opts))
      # Note.: compensating dynamic range mismatch between input features and Sigmoid-hidden layers,
      # i.e. mapping the std-dev of N(0,1) (input features) to std-dev of U[0,1] (sigmoid-outputs).
      # This is done by multiplying with stddev(U[0,1]) = sqrt(1/12).
      # The stddev of weights is consequently reduced with scale 0.29,
print("%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts))
if o.with_dropout:
  print("<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts))


# Internal AffineTransforms,
for i in range(num_hid_layers-1):
  print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
        (num_hid_neurons, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
         (o.param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons)), o.max_norm, o.affine_opts))
  print("%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts))
  if o.with_dropout:
    print("<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts))

# Optionaly add bottleneck,
if o.bottleneck_dim != 0:
  assert(o.bottleneck_dim > 0)
  if o.bottleneck_trick:
    # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
    print("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f" % \
     (num_hid_neurons, o.bottleneck_dim, \
      (o.param_stddev_factor * Glorot(num_hid_neurons, o.bottleneck_dim) * 0.75 ), 0.1))
    # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
    print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f <MaxNorm> %f %s" % \
     (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
      (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1, o.max_norm, o.affine_opts))
  else:
    # Same learninig-rate and stddev-formula everywhere,
    print("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f" % \
     (num_hid_neurons, o.bottleneck_dim, \
      (o.param_stddev_factor * Glorot(num_hid_neurons, o.bottleneck_dim))))
    print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <MaxNorm> %f %s" % \
     (o.bottleneck_dim, num_hid_neurons, o.hid_bias_mean, o.hid_bias_range, \
      (o.param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons)), o.max_norm, o.affine_opts))
  print("%s <InputDim> %d <OutputDim> %d %s" % (o.activation_type, num_hid_neurons, num_hid_neurons, o.activation_opts))
  if o.with_dropout:
    print("<Dropout> <InputDim> %d <OutputDim> %d %s" % (num_hid_neurons, num_hid_neurons, o.dropout_opts))

# Last AffineTransform (10x smaller learning rate on bias)
print("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f" % \
      (num_hid_neurons, num_leaves, 0.0, 0.0, \
       (o.param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1))

# Optionaly append softmax
if o.with_softmax:
  if o.block_softmax_dims == "":
    print("<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves))
  else:
    print("<BlockSoftmax> <InputDim> %d <OutputDim> %d <BlockDims> %s" % (num_leaves, num_leaves, o.block_softmax_dims))

# We are done!
sys.exit(0)

