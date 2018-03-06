#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
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


# This script takes three command-line arguments.
# The first is a log-file such as exp/tri4b_nnet/log/combine.10.log,
# which is the output of nnet-combine.  The second is a file such
# as exp/tri4b_nnet/11.tmp.mdl, i.e. a model file, for which we will
# update the learning rates; the third is the output nnet file e.g.
# exp/tri4b_nnet/11.mdl

# This script assumes that the "combine" script is called as:
# nnet-combine <old-model> <new-model-1> <new-model-2> ... <new-model-n> <validation-examples> <output-model>.
# It gets from the logging output a line like this:
# LOG (nnet-combine:CombineNnets():combine-nnet.cc:184) Combining nnets, validation objf per frame changed from -1.43424 to -1.42067, scale factors are  [ 0.727508 0.79889 0.299533 0.137696 -0.0479123 0.210445 0.0195638 0.123843 0.167453 0.0193894 -0.0128672 0.178384 0.0516549 0.0958205 0.125495 ]
# [in this case the 1st 3 numbers correspond to the <old-model> ] and for each
# updatable layer, it works out the total weight on the new models.
# It interprets this as being (for each layer) a step length along
# the path old-model -> new-model.
# Basically, we change the learning rate by a factor equal to this step length,
# subject to limits on the change  [by default limit to halving/doubling].
# It's fairly obvious why we would want do do this.

# These options can be useful if we want to splice the input
# features across time.
$sources_to_exclude = 1; # may make this configurable later.
$min_learning_rate_factor = 0.5;
$max_learning_rate_factor = 2.0;
$min_learning_rate = 0.0001; # Put a floor because if too small,
  # the changes become zero due to roundoff.

if (@ARGV > 0) {
  for ($x = 1; $x < 10; $x++) {
    if ($ARGV[0] eq "--min-learning-rate-factor") {
      $min_learning_rate_factor = $ARGV[1];
      shift; shift;
    }
    if ($ARGV[0] eq "--max-learning-rate-factor") {
      $max_learning_rate_factor = $ARGV[1];
      shift; shift;
    }
    if ($ARGV[0] eq "--min-learning-rate") {
      $min_learning_rate = $ARGV[1];
      shift; shift;
    }
  }
}


if (@ARGV != 3) {
  print STDERR "Usage: update_learning_rates.pl [options] <log-file-for-nnet-combine> <nnet-in> <nnet-out>
Options:
   --min-learning-rate-factor       #  minimum factor to change learning rate by (default: 0.5)
   --max-learning-rate-factor       #  maximum factor to change learning rate by (default: 2.0)\n";
   exit(1);
}

($combine_log, $nnet_in, $nnet_out) = @ARGV;

open(L, "<$combine_log") || die "Opening log file \"$combine_log\"";


while(<L>) {
  if (m/Objective functions for the source neural nets are\s+\[(.+)\]/) {
    ## a line like:
    ##  LOG (nnet-combine:GetInitialScaleParams():combine-nnet.cc:66) Objective functions for the source neural nets are  [ -1.37002 -1.52115 -1.52103 -1.50189 -1.51912 ]
    @A = split(" ", $1);
    $num_sources = @A; # number of source neural nets (dimension of @A); 5 in this case.
  }
  ## a line like:
  ## LOG (nnet-combine:CombineNnets():combine-nnet.cc:184) Combining nnets, validation objf per frame changed from -1.37002 to -1.36574, scale factors are  [ 0.819379 0.696122 0.458798 0.040513 -0.0448875 0.171431 0.0274615 0.139143 0.133846 0.0372585 0.114193 0.17944 0.0491838 0.0668778 0.0328936 ]
  if (m/Combining nnets.+scale factors are\s+\[(.+)\]/) {
    @scale_factors = split(" ", $1);
  }
}

if (!defined $num_sources) {
  die "Log file $combine_log did not have expected format: no line with \"Objective functions\"\n";
}
if (!defined @scale_factors) {
  die "Log file $combine_log did not have expected format: no line with \"Combining nnets\"\n";
}


$num_scales = @scale_factors; # length of the array.
if ($num_scales % $num_sources != 0) {
  die "Error interpreting log file $combine_log: $num_sources does not divide $num_scales\n";
}
close(L);

open(P, "nnet-am-info $nnet_in |") || die "Opening pipe from nnet-am-info";
@learning_rates = ();
while(<P>) {
  if (m/learning rate = ([^,]+),/) {
    push @learning_rates, $1;
  }
}
close(P);

$num_layers = $num_scales / $num_sources;

$num_info_learning_rates = @learning_rates;
if ($num_layers != $num_info_learning_rates) {
  die "From log file we expect there to be $num_layers updatable components, but from the output of nnet-am-info we saw $num_info_learning_rates";
}

for ($layer = 0; $layer < $num_layers; $layer++) {
  # getting the sum of the weights for this layer from all the non-excluded sources.
  $sum = 0.0;
  for ($source = $sources_to_exclude; $source < $num_sources; $source++) {
    $index = ($source * $num_layers) + $layer;
    $sum += $scale_factors[$index];
  }
  $learning_rate_factor = $sum;
  if ($learning_rate_factor > $max_learning_rate_factor) { $learning_rate_factor = $max_learning_rate_factor; }
  if ($learning_rate_factor < $min_learning_rate_factor) { $learning_rate_factor = $min_learning_rate_factor; }
  $old_learning_rate = $learning_rates[$layer];
  $new_learning_rate = $old_learning_rate * $learning_rate_factor;
  if ($new_learning_rate < $min_learning_rate) { $new_learning_rate = $min_learning_rate; }
  print STDERR "For layer $layer, sum of weights of non-excluded sources is $sum, learning-rate factor is $learning_rate_factor\n";
  $learning_rates[$layer] = $new_learning_rate;
}

$lrates_string=join(":", @learning_rates);

$ret = system("nnet-am-copy --learning-rates=$lrates_string $nnet_in $nnet_out");

exit($ret != 0);
