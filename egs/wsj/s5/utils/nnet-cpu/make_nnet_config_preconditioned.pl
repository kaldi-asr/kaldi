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
$single_layer_config = ""; # a file to which we'll output a config corresponding
       # to a single layer; we'll later use this to add layers to the neural
       # network.
$bias_stddev = 2.0;  # Standard deviation for random initialization of the
                     # bias terms (mean is zero).
$splice_max_context = 0; # Relates to SpliceMaxComponent (experimental feature)
$learning_rate = 0.001;
$max_change = 0.0;
$nonlinear_component_type = "Tanh";

$alpha = 4.0;
$l2_penalty_opt = ""; # Option for AffineComponentPreconditioned layer.
$tree_map = ""; # If supplied, a text file that maps from l2 to l1 tree nodes (output
   # by build-tree-two-level).  Used for initializing mixture-prob component.

$splice_context = 0;
$dropout_scale = -1.0; # if not -1.0, scale for "lower" part of 
                       # dropout scale, typically 0 <= dropout_scale < 1.
$additive_noise_stddev = 0.0; # I didn't find this helpful either.
$lda_dim = 0;
$expand_power = 1;
$expand_scale = 1.0;
$lda_mat = "";

for ($x = 1; $x < 10; $x++) {
  if ($ARGV[0] eq "--input-left-context") {
    $input_left_context = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--l2-penalty") {
    my $l2_penalty = $ARGV[1];
    $l2_penalty_opt = "l2-penalty=$l2_penalty";
    shift; shift;
  }
  if ($ARGV[0] eq "--dropout-scale") {
    $dropout_scale = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--expand-power") {
    $expand_power = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--expand-scale") {
    $expand_scale = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--max-change") {
    $max_change = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--additive-noise-stddev") {
    $additive_noise_stddev = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--nonlinear-component-type") {
    $nonlinear_component_type = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--lda-mat") {
    $splice_context = $ARGV[1];
    $lda_dim = $ARGV[2];
    $lda_mat = $ARGV[3];
    shift; shift; shift; shift;
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
  if ($ARGV[0] eq "--alpha") {
    $alpha = $ARGV[1];
    shift; shift;
  }
  if ($ARGV[0] eq "--splice-max-context") {
    $splice_max_context = $ARGV[1];
    shift; shift;
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
  if ($ARGV[0] eq "--tree-map") { # Note: this was for an idea that
    # didn't end up working for me; it relates to SCTM-like systems.
    $tree_map = $ARGV[1];
    shift; shift;
  }
}


if (@ARGV != 4) {
  print STDERR "Usage: make_nnet_config_preconditioned.pl  [options] <feat-dim> <num-leaves> <num-hidden-layers> <num-parameters>  >config-file
Options:
   --input-left-context <n>        #  #frames of left context for input features; default 0 (this separate from pre-LDA splicing).
   --input-right-context <n>       #  #frames of right context for input features; default 0  (this separate from pre-LDA splicing).
   --param-stdddev-factor <f>      #  Factor which can be used to modify the standard deviation of
                                   #  randomly nitialized features (default, 1.  Gets multiplied by
                                   #  1/sqrt of number of inputs).
   --initial-num-hidden-layers <n> <config-file>   #  If >0, number of hidden layers to initialize the network with.
                                   #  In this case, the positional parameter <num-hidden-layers> is only
                                   #  used to work out the number of units per hidden layer (based on
                                   #  parameter count), and we write to <config-file> the config corresponding
                                   #  to a single hidden layer.
   --alpha <f>                     #  Factor (default 0.1) which affects the preconditioning.  0 < alpha <= 1;
                                   #  smaller means more aggressive preconditioning / less smoothing of the Fisher
                                   #  matrix.
   --learning-rate <f>             # Initial learning rate, default 0.001
   --lda-mat <splice-width> <lda-dimension> <lda-matrix-filename>  # Allows the user to specify splice-and-lda
                                   # with a given transformation, as a fixed component in the network.  E.g.
                                   # splice-width of 4 represents context of +- 4 frames.  Here, lda-dimension is
                                   # the output dimension of LDA, which must be the same as in the file.\n";
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
##            + hidden_layer_size * (num_leaves + feat_dim * context_size * expand_power)
## solve for hidden_layer_size = x.
## a x^2 + b x + c, with
## a = num_hidden_layers - 1
## b = num_leaves + feat_dim * context_size
## c = -num_params

$a = $num_hidden_layers - 1;
$b = $num_leaves + $feat_dim * $context_size * $expand_power;
$c = -$num_params;

if ($a > 0) {
  $hidden_layer_size =  int((-$b + sqrt($b*$b - 4*$a*$c)) / (2*$a));
} else {
  $hidden_layer_size = int(-$c/$b);
}


$actual_num_params = $hidden_layer_size * $hidden_layer_size * ($num_hidden_layers - 1)
                   + $hidden_layer_size * ($num_leaves + $feat_dim * $context_size * $expand_power);

if (abs($actual_num_params - $num_params) > 0.1 * $num_params) {
  print STDERR "Warning: make_nnet_config.pl: possible failure $actual_num_params != $num_params";
}

if ($splice_context > 0) { # --lda-mat <splice-context> <lda-matrix> was specified...
  print "SpliceComponent input-dim=$feat_dim left-context=$splice_context right-context=$splice_context\n";
  print "FixedLinearComponent matrix=$lda_mat\n"; # specify the filename.
  $feat_dim = $lda_dim; # This is now the input dimension.
}

if ($splice_max_context > 0) {
  print "SpliceMaxComponent dim=$feat_dim left-context=$splice_max_context right-context=$splice_max_context\n";
}


if ($input_left_context + $input_right_context != 0) {
  # First component has to be splicing component...
  # Note: we might be interested in decorrelating this e.g. with
  # DCT layer at some point, but for now, splicing isn't seeming to be
  # that useful.
  print "SpliceComponent input-dim=$feat_dim left-context=$input_left_context right-context=$input_right_context\n";
}
$cur_input_dim = $feat_dim * (1 + $input_left_context + $input_right_context);

if ($expand_power > 1) {
  print "PowerExpandComponent input-dim=$cur_input_dim max-power=$expand_power higher-power-scale=$expand_scale\n";
  $cur_input_dim *= $expand_power;
}

for ($hidden_layer = 0; $hidden_layer < $initial_num_hidden_layers; $hidden_layer++) {
  $param_stddev = $param_stddev_factor * 1.0 / sqrt($cur_input_dim);
  print "AffineComponentPreconditioned input-dim=$cur_input_dim output-dim=$hidden_layer_size alpha=$alpha max-change=$max_change " .
    "$l2_penalty_opt learning-rate=$learning_rate param-stddev=$param_stddev bias-stddev=$bias_stddev\n";
  $cur_input_dim = $hidden_layer_size;
  print "${nonlinear_component_type}Component dim=$cur_input_dim\n";
  if ($dropout_scale != -1.0) {
    print "DropoutComponent dim=$cur_input_dim dropout-scale=$dropout_scale\n";
  }
  if ($additive_noise_stddev != 0.0) {
    print "AdditiveNoiseComponent dim=$cur_input_dim stddev=$additive_noise_stddev\n";
  }
}

if ($single_layer_config ne "") {
  # Create a config file we'll use to add new hidden layers.
  open(F, ">$single_layer_config") || die "Error opening $single_layer_config for output";
  $param_stddev = $param_stddev_factor * 1.0 / sqrt($hidden_layer_size);
  print F "AffineComponentPreconditioned input-dim=$hidden_layer_size output-dim=$hidden_layer_size alpha=$alpha max-change=$max_change " .
    "$l2_penalty_opt learning-rate=$learning_rate param-stddev=$param_stddev bias-stddev=$bias_stddev\n";
  print F "${nonlinear_component_type}Component dim=$hidden_layer_size\n";
  if ($dropout_scale != -1.0) {
    print F "DropoutComponent dim=$cur_input_dim dropout-scale=$dropout_scale\n";
  }
  if ($additive_noise_stddev != 0.0) {
    print F "AdditiveNoiseComponent dim=$cur_input_dim stddev=$additive_noise_stddev\n";
  }
  close (F) || die "Closing config file";
}

## Now the output layer.
print "AffineComponentPreconditioned input-dim=$cur_input_dim output-dim=$num_leaves alpha=$alpha max-change=$max_change " .
  "$l2_penalty_opt learning-rate=$learning_rate param-stddev=0 bias-stddev=0\n"; # we just set the parameters to zero for this layer.
## the softmax nonlinearity.
print "SoftmaxComponent dim=$num_leaves\n";

if ($tree_map ne "") {
  # Create a MixtureProbComponent at the end, that shares "Gaussians"
  # among leaves that share the same level-1 tree index.
  open(F, "<$tree_map") || die "opening tree map file $tree_map";
  $map = <F>;
  close(F);
  $map =~ s/\s*\[\s*// || die "Unexpected data in tree map file $tree_map";
  $map =~ s/\s*\]\s*// || die "Unexpected data in tree map file $tree_map";
  @map = split(" ", $map);
  @dims = ();
  while (@map > 0) {
    $index = shift @map;
    $n = 1;
    while (@map > 0 && $map[0] == $index) { shift @map; $n++; }
    push @dims, $n;
  }
  $dims = join(":", @dims);
  print "MixtureProbComponent learning-rate=$learning_rate diag-element=0.9 dims=$dims\n";
}

##
