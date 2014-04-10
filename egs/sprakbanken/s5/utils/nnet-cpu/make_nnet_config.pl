#!/usr/bin/perl -w
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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


# These options can be useful if we want to splice the input
# features across time.
$input_left_context = 0;
$input_right_context = 0;
$param_stddev_factor = 1.0;  # can be used to adjust initial variance
  # of parameters.
$initial_num_hidden_layers = -1; # if >= 0, the number of hidden layers
  # the model should start with, which may be less than the final number
  # (the final number is used to calculate the #neurons).
$single_layer_config = "";
$bias_stddev = 2.0;
$learning_rate = 0.001;
$nobias = "";

for ($x = 1; $x < 10; $x++) {
  if ($ARGV[0] eq "--input-left-context") {
    $input_left_context = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--input-right-context") {
    $input_right_context = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--param-stddev-factor") {
    $param_stddev_factor = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--bias-stddev") {
    $bias_stddev = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--nobias") {
    $nobias = "Nobias";
    shift;
  }
  if ($ARGV[0] eq "--learning-rate") {
    $learning_rate = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--initial-num-hidden-layers") {
    $initial_num_hidden_layers = $ARGV[1];
    $single_layer_config = $ARGV[2];
    shift; shift; shift;
  }
}


if (@ARGV != 4) {
  print STDERR "Usage: make_nnet_config.pl  [options] <feat-dim> <num-leaves> <num-hidden-layers> <num-parameters>  >config-file
Options:
   --input-left-context <n>        #  #frames of left context for input features; default 0.
   --input-right-context <n>       #  #frames of right context for input features; default 0.
   --param-stdddev-factor <f>      #  Factor which can be used to modify the standard deviation of
                                   #  randomly initialized features (default, 1.  Gets multiplied by
                                   #  1/sqrt of number of inputs).
   --initial-num-hidden-layers <n> <config-file>   #  If >0, number of hidden layers to initialize the network with.
                                   #  In this case, the positional parameter <num-hidden-layers> is only
                                   #  used to work out the number of units per hidden layer (based on
                                   #  parameter count), and we write to <config-file> the config corresponding
                                   #  to a single hidden layer.
   --learning-rate <f>             # Initial learning rate, default 0.001\n";
     exit(1);
}

($feat_dim, $num_leaves, $num_hidden_layers, $num_params) = @ARGV;
($input_left_context < 0) &&  die "Invalid input left context $input_left_context";
($input_right_context < 0) &&  die "Invalid input right context $input_right_context";
($feat_dim <= 0) &&  die "Invalid feature dimension $feat_dim";
($num_leaves <= 0) && die "Invalid number of leaves $num_leaves";
($num_hidden_layers <= 0) && die "Invalid number of hidden layers $num_hidden_layers";
if ($initial_num_hidden_layers < 0) {
  $initial_num_hidden_layers = $num_hidden_layers;
}
if ($initial_num_hidden_layers > $num_hidden_layers) {
  print STDERR "Initial number of hidden layers is more than #hidden layers.\n" .
    "This does not really make sense but continuing anyway.";
}

$context_size = 1 + $input_left_context + $input_right_context;
($num_params < ($num_leaves + ($feat_dim * $context_size) + $num_hidden_layers + 1))
  && die "Invalid number of params $num_params";

## num_params = hidden_layer_size^2 * (num_hidden_layers-1)
##            + hidden_layer_size * (num_leaves + feat_dim * context_size)
## solve for hidden_layer_size = x.
## a x^2 + b x + c, with
## a = num_hidden_layers - 1
## b = num_leaves + feat_dim * context_size
## c = -num_params

$a = $num_hidden_layers - 1;
$b = $num_leaves + $feat_dim * $context_size;
$c = -$num_params;

if ($a > 0) {
  $hidden_layer_size =  int((-$b + sqrt($b*$b - 4*$a*$c)) / (2*$a));
} else {
  $hidden_layer_size = int(-$c/$b);
}


$actual_num_params = $hidden_layer_size * $hidden_layer_size * ($num_hidden_layers - 1)
                   + $hidden_layer_size * ($num_leaves + $feat_dim * $context_size);

if (abs($actual_num_params - $num_params) > 0.1 * $num_params) {
  print STDERR "Warning: make_nnet_config.pl: possible failure $actual_num_params != $num_params";
}

if ($input_left_context + $input_right_context != 0) {
  # First component has to be splicing component...
  # Note: we might be interested in decorrelating this e.g. with
  # DCT layer at some point, but for now, splicing isn't seeming to be
  # that useful.
  print "SpliceComponent input-dim=$feat_dim left-context=$input_left_context right-context=$input_right_context\n";
}
$cur_input_dim = $feat_dim * (1 + $input_left_context + $input_right_context);

for ($hidden_layer = 0; $hidden_layer < $initial_num_hidden_layers; $hidden_layer++) {
  $param_stddev = $param_stddev_factor * 1.0 / sqrt($cur_input_dim);
  print "AffineComponent$nobias input-dim=$cur_input_dim output-dim=$hidden_layer_size " .
    "learning-rate=$learning_rate param-stddev=$param_stddev bias-stddev=$bias_stddev\n";
  $cur_input_dim = $hidden_layer_size;
  print "TanhComponent dim=$cur_input_dim\n";
}

if ($single_layer_config ne "") {
  # Create a config file we'll use to add new hidden layers.
  open(F, ">$single_layer_config") || die "Error opening $single_layer_config for output";
  $param_stddev = $param_stddev_factor * 1.0 / sqrt($hidden_layer_size);
  print F "AffineComponent$nobias input-dim=$hidden_layer_size output-dim=$hidden_layer_size " .
    "learning-rate=$learning_rate param-stddev=$param_stddev bias-stddev=$bias_stddev\n";
  print F "TanhComponent dim=$hidden_layer_size\n";
  close (F) || die "Closing config file";
}

## Now the output layer.
print "AffineComponent$nobias input-dim=$cur_input_dim output-dim=$num_leaves " .
  "learning-rate=$learning_rate param-stddev=0 bias-stddev=0\n"; # we just set the parameters to zero for this layer.
## the softmax nonlinearity.
print "SoftmaxComponent dim=$num_leaves\n";

##
