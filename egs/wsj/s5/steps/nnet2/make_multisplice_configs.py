#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey and Vijayaditya Peddinti).  Apache 2.0.

# Creates the nnet.config and hidde_*.config scripts used in train_pnorm_multisplice.sh
# Parses the splice string to generate relevant variables for get_egs.sh, get_lda.sh and nnet/hidden.config files

from __future__ import division
from __future__ import print_function
import re, argparse, sys, math, warnings

# returns the set of frame indices required to perform the convolution
# between sequences with frame indices in x and y
def get_convolution_index_set(x, y):
  z = []
  for i in range(len(x)):
    for j in range(len(y)):
      z.append(x[i]+y[j])
  z = list(set(z))
  z.sort()
  return z

def parse_splice_string(splice_string):
  layerwise_splice_indexes = splice_string.split('layer')[1:]
  print(splice_string.split('layer'))
  contexts={}
  first_right_context = 0 # default value
  first_left_context = 0 # default value
  nnet_frame_indexes = [0] # frame indexes required by the network
                           # at the initial layer (will be used in 
                           # determining the context for get_egs.sh)
  try:
    for cur_splice_indexes in layerwise_splice_indexes:
      layer_index, frame_indexes  = cur_splice_indexes.split("/")
      frame_indexes = [int(x) for x in frame_indexes.split(':')]
      layer_index = int(layer_index)
      assert(layer_index >= 0)
      if layer_index == 0:
        first_left_context = min(frame_indexes)
        first_right_context = max(frame_indexes)
        try:
          assert(frame_indexes == list(range(first_left_context, first_right_context+1)))
        except AssertionError:
          raise Exception('Currently the first splice component just accepts contiguous context.')
        try:
          assert((first_left_context <=0) and (first_right_context >=0))
        except AssertionError:
          raise Exception("""get_lda.sh script does not support postive left-context or negative right context.
          left context provided is %d and right context provided is %d.""" % (first_left_context, first_right_context))
        # convolve the current splice indices with the splice indices until last layer
      nnet_frame_indexes = get_convolution_index_set(frame_indexes, nnet_frame_indexes)
      cur_context = ":".join([str(x) for x in frame_indexes])
      contexts[layer_index] = cur_context
  except ValueError:
    raise Exception('Unknown format in splice_indexes variable: {0}'.format(params.splice_indexes))
  print(nnet_frame_indexes)
  max_left_context = min(nnet_frame_indexes)
  max_right_context = max(nnet_frame_indexes)
  return [contexts, ' nnet_left_context={0};\n nnet_right_context={1}\n first_left_context={2};\n first_right_context={3}\n'.format(abs(max_left_context), abs(max_right_context), abs(first_left_context), abs(first_right_context) )]

def create_config_files(output_dir, params):
  pnorm_p = 2
  pnorm_input_dim = params.pnorm_input_dim
  pnorm_output_dim = params.pnorm_output_dim
  contexts, context_variables = parse_splice_string(params.splice_indexes)
  var_file = open("{0}/vars".format(output_dir), "w")
  var_file.write(context_variables)
  var_file.close()

  try:
    assert(max(contexts.keys()) < params.num_hidden_layers)
  except AssertionError:
    raise Exception("""Splice string provided is {2}.
    Number of hidden layers {0}, is less than the number of context specifications provided.
    Splicing is supported only until layer {1}.""".format(params.num_hidden_layers, params.num_hidden_layers - 1, params.splice_indexes))

  stddev=1.0/math.sqrt(pnorm_input_dim)
  try :
    nnet_config = ["SpliceComponent input-dim={0} context={1} const-component-dim={2}".format(params.total_input_dim, contexts[0], params.ivector_dim),
    "FixedAffineComponent matrix={0}".format(params.lda_mat),
    "AffineComponentPreconditionedOnline input-dim={0} output-dim={1} {2} learning-rate={3} param-stddev={4} bias-stddev={5}".format(params.lda_dim, pnorm_input_dim, params.online_preconditioning_opts, params.initial_learning_rate, stddev, params.bias_stddev),
    ("PnormComponent input-dim={0} output-dim={1} p={2}".format(pnorm_input_dim, pnorm_output_dim, pnorm_p) if pnorm_input_dim != pnorm_output_dim else "RectifiedLinearComponent dim={0}".format(pnorm_input_dim)),
    "NormalizeComponent dim={0}".format(pnorm_output_dim),
    "AffineComponentPreconditionedOnline input-dim={0} output-dim={1} {2} learning-rate={3} param-stddev=0 bias-stddev=0".format(pnorm_output_dim, params.num_targets, params.online_preconditioning_opts, params.initial_learning_rate),
    "SoftmaxComponent dim={0}".format(params.num_targets)]

    nnet_config_file = open(("{0}/nnet.config").format(output_dir), "w")
    nnet_config_file.write("\n".join(nnet_config))
    nnet_config_file.close()
  except KeyError:
    raise Exception('A splice layer is expected to be the first layer. Provide a context for the first layer.')

  for i in range(1, params.num_hidden_layers): #just run till num_hidden_layers-1 since we do not add splice before the final affine transform
    lines=[]
    context_len = 1
    if i in contexts:
        # Adding the splice component as a context is provided
        lines.append("SpliceComponent input-dim=%d context=%s " % (pnorm_output_dim, contexts[i]))
        context_len = len(contexts[i].split(":"))
    # Add the hidden layer, which is a composition of an affine component, pnorm component and normalization component
    lines.append("AffineComponentPreconditionedOnline input-dim=%d output-dim=%d %s learning-rate=%f param-stddev=%f bias-stddev=%f" 
        % ( pnorm_output_dim*context_len, pnorm_input_dim, params.online_preconditioning_opts, params.initial_learning_rate, stddev, params.bias_stddev))
    if pnorm_input_dim != pnorm_output_dim:
      lines.append("PnormComponent input-dim=%d output-dim=%d p=%d" % (pnorm_input_dim, pnorm_output_dim, pnorm_p))
    else:
      lines.append("RectifiedLinearComponent dim=%d" % (pnorm_input_dim)) 
      warnings.warn("Using the RectifiedLinearComponent, in place of the PnormComponent as pnorm_input_dim == pnorm_output_dim")
    lines.append("NormalizeComponent dim={0}".format(pnorm_output_dim))
    out_file = open("{0}/hidden_{1}.config".format(output_dir, i), 'w')
    out_file.write("\n".join(lines))
    out_file.close()


if __name__ == "__main__":
  print(" ".join(sys.argv))
  parser = argparse.ArgumentParser()
  parser.add_argument('--splice-indexes', type=str, help='string specifying the indexes for the splice layers throughout the network')
  parser.add_argument('--total-input-dim', type=int, help='dimension of the input to the network')
  parser.add_argument('--ivector-dim', type=int, help='dimension of the ivector portion of the neural network input')
  parser.add_argument('--lda-mat', type=str, help='lda-matrix used after the first splice component')
  parser.add_argument('--lda-dim', type=str, help='dimension of the lda output')
  parser.add_argument('--pnorm-input-dim', type=int, help='dimension of input to pnorm layer')
  parser.add_argument('--pnorm-output-dim', type=int, help='dimension of output of pnorm layer')
  parser.add_argument('--online-preconditioning-opts', type=str, help='extra options for the AffineComponentPreconditionedOnline component')
  parser.add_argument('--initial-learning-rate', type=float, help='')
  parser.add_argument('--num-targets', type=int, help='#targets for the neural network ')
  parser.add_argument('--num-hidden-layers', type=int, help='#hidden layers in the neural network ')
  parser.add_argument('--bias-stddev', type=float, help='standard deviation of r.v. used for bias component initialization')
  parser.add_argument("mode", type=str, help="contexts|configs")
  parser.add_argument("output_dir", type=str, help="output directory to store the files")
  params = parser.parse_args() 
  
  print(params)
  if params.mode == "contexts":
    [context, context_variables] = parse_splice_string(params.splice_indexes)
    var_file = open("{0}/vars".format(params.output_dir), "w")
    var_file.write(context_variables)
    var_file.close()
  elif params.mode == "configs":
    create_config_files(params.output_dir, params)
  else:
    raise Exception("mode has to be in the set {contexts, configs}")
