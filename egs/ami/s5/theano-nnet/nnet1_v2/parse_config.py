#!/usr/bin/env python

import os, errno, sys

def parse_config_dict(config_file, d_args=[]):
  lines = map(lambda x: x.strip(), open(config_file, "r").readlines())
  for line in lines:
    line = line.lstrip() 
    if line[0:2] == "--":
      line = line.split("#")[0]
      line = line.strip()
      d_args.append(line)
  return d_args

def parse_config(parser, config_file):

  args = sys.argv[1:]
  args = parse_config_dict(config_file, args)
  return parser.parse_args(args)



def feature_preprocess_options(parser):
  # parser is OptionParser object

  # delta options
  parser.add_option('--delta-order', dest="delta_order",
                    help="Order of delta computation [default: %default]",
                    default=0, type=int)
  parser.add_option('--delta-window', dest="delta_window",
                    help="Parameter controlling window for delta computation (actual window size for each delta order is 1 + 2*delta-window-size) [default: %default]",
                    default=2, type=int)
  
  # splice options
  parser.add_option('--splice-vec', dest="splice_vec",
                    help="splice vector. kaldi format [default: %default]",
                    default="", type=str)
  parser.add_option('--splice', dest="splice",
                    help="splice [default: %default]",
                    default=0, type=int)
  parser.add_option('--splice-step', dest="splice_step",
                    help="splice step [default: %default]",
                    default=1, type=int)
  
  # apply cmvn options
  parser.add_option('--norm-means', dest='norm_means', 
                    help='apply speaker cmn',
                    default="false", type=str)
  parser.add_option('--norm-vars', dest='norm_vars', 
                    help='apply speaker cvn',
                    default="false", type=str)

  parser.add_option('--trn-utt2spk-file', dest="trn_utt2spk_file",
                    help="kaldi train utt2spk [default: %default]",
                    default="", type=str)
  parser.add_option('--trn-cmvn-scp', dest="trn_cmvn_scp",
                    help="kaldi train utt2spk [default: %default]",
                    default="", type=str)

  parser.add_option('--cv-utt2spk-file', dest="cv_utt2spk_file",
                    help="kaldi cv utt2spk [default: %default]",
                    default="", type=str)
  parser.add_option('--cv-cmvn-scp', dest="cv_cmvn_scp",
                    help="kaldi cv utt2spk [default: %default]",
                    default="", type=str)

  parser.add_option('--utt2spk-file', dest="utt2spk_file",
                    help="kaldi utt2spk [default: %default]",
                    default="", type=str)
  parser.add_option('--cmvn-scp', dest="cmvn_scp",
                    help="kaldi utt2spk [default: %default]",
                    default="", type=str)


  # TODO: Other stuff
  

  return parser


def nnet_arch_options(parser):

  parser.add_option('--nnet-arch', dest="nnet_arch",
                    help="[default: %default]",
                    default=None, type=str)
  parser.add_option('--nnet-layers-types', dest="nnet_layers_types",
                    help="[default: %default]",
                    default="", type=str)

  parser.add_option('--hid-layer-sizes', dest="hid_layer_sizes",
                    help="[default: %default]",
                    default=None, type=str)

  parser.add_option('--nnet-proto', dest="nnet_proto",
                    help="[default: %default]",
                    default=None, type=str)
    
  parser.add_option('--num-hid-layers', dest="num_hid_layers",
                    help="[default: %default]",
                    default=4, type=int)
  parser.add_option('--hid-dim', dest="hid_dim",
                    help="[default: %default]",
                    default=1024, type=int)
  parser.add_option('--bn-dim', dest="bn_dim",
                    help="[default: %default]",
                    default=0, type=int)

  parser.add_option('--inp-dim', dest="inp_dim", help="[default: %default]", default=0, type=int)
  parser.add_option('--num-tgts', dest="num_tgts", help="[default: %default]", default=0, type=int)


  parser.add_option('--no-softmax', dest='with_softmax', help='Do not put <SoftMax> in the prototype [default: %default]', default=True, action='store_false');
  parser.add_option('--hid-bias-mean', dest='hid_bias_mean', help='Set bias for hidden activations [default: %default]', default=-2.0, type='float');
  parser.add_option('--hid-bias-range', dest='hid_bias_range', help='Set bias range for hidden activations (+/- 1/2 range around mean) [default: %default]', default=4.0, type='float');
  parser.add_option('--param-stddev-factor', dest='param_stddev_factor', help='Factor to rescale Normal distriburtion for initalizing weight matrices [default: %default]', default=0.1, type='float');

  parser.add_option('--no-glorot-scaled-stddev', dest='with_glorot', help='Generate normalized weights according to X.Glorot paper, but mapping U->N with same variance (factor sqrt(x/(dim_in+dim_out)))', action='store_false', default=True);
  parser.add_option('--no-smaller-input-weights', dest='smaller_input_weights', help='Disable 1/12 reduction of stddef in input layer [default: %default]', action='store_false', default=True);
  parser.add_option('--max-norm', dest='max_norm', help='Max radius of neuron-weights in L2 space (if longer weights get shrinked, not applied to last layer, 0.0 = disable) [default: %default]', default=0.0, type='float');


  return parser



def nnet_hyperparams_options(parser):


  parser.add_option('--learn-rate', dest="learn_rate", help="[default: %default]", default=0.008, type=float)

  parser.add_option('--keep-lr-iters', dest="keep_lr_iters", help="fix learning rate for N initial epochs [default: %default]", default=0, type=int)
  parser.add_option('--start-learn-rate', dest="start_learn_rate", help="[default: %default]", default=0.008, type=float)
  parser.add_option('--end-learn-rate', dest="end_learn_rate", help="[default: %default]", default=0.008/512, type=float)
  parser.add_option('--decay-factor', dest="decay_factor", help="[default: %default]", default=1e8, type=float) #9iters on 80 hrs of speech
  

  parser.add_option('--segment-buffer-size', dest="segment_buffer_size", help="[default: %default]", default=80, type=int)
  parser.add_option('--batch-size', dest="batch_size", help="[default: %default]", default=256, type=int)

  parser.add_option('--max-iters', dest="max_iters", help="[default: %default]", default=20, type=int)
  parser.add_option('--min-iters', dest="min_iters", help="keep training, disable weight rejection, start learn-rate halving as usual, [default: %default]", default=0, type=int)

  parser.add_option('--tolerance', dest="tolerance", help="[default: %default]", default=0.003, type=float)

  parser.add_option('--start-halving-impr', dest="start_halving_impr", help="[default: %default]", default=0.01, type=float)
  parser.add_option('--end-halving-impr', dest="end_halving_impr", help="[default: %default]", default=0.001, type=float)
  parser.add_option('--halving-factor', dest="halving_factor", help="[default: %default]", default=0.5, type=float)

  return parser

def nnet_donefile_options(parser):

  parser.add_option('--train-error', dest="train_error", help="[default: %default]", default=1.0, type=float)
  parser.add_option('--train-accuracy', dest="train_accuracy", help="[default: %default]", default=1.0, type=float)
  parser.add_option('--cv-error', dest="cv_error", help="[default: %default]", default=1.0, type=float)
  parser.add_option('--cv-accuracy', dest="cv_accuracy", help="[default: %default]", default=1.0, type=float)
  parser.add_option('--learn-rate', dest="learn_rate", help="[default: %default]", default=0.008, type=float)
  return parser


def theano_nnet_parse_opts(parser):
  
  parser = feature_preprocess_options(parser)
  parser = nnet_arch_options(parser)
  parser = nnet_hyperparams_options(parser)

  return parser

