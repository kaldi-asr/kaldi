#!/usr/bin/env bash

# Copyright 2014    Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script converts nnet2 models which expect splice+LDA as the input, into
# models which expect raw features (e.g. MFCC) as the input.  If you include
# the option --global-cmvn-stats <matrix>, it will also remove CMVN from the model
# by including it as part of the neural net.


# Begin configuration section
cleanup=true
global_cmvn_stats=
cmd=run.pl
# learning_rate and max_change will only make a difference if we train this model, which is unlikely.
learning_rate=0.00001 # give it a tiny learning rate by default; the user
                      # should probably tune this or set it if they want to train.
max_change=5.0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <src-nnet-dir> <dest-nnet-dir>"
  echo "e.g.: $0 --global-cmvn-stats global_cmvn.mat exp/dnn4b_nnet2 exp/dnn4b_nnet2_raw"
  echo "Options include"
  echo "   --global-cmvn-stats <stats-file>         # Filename of globally summed CMVN stats, if"
  echo "                                            # you want to push the CMVN inside the nnet"
  echo "                                            # (it won't any longer be speaker specific)"
  exit 1;
fi

src=$1
dir=$2

mkdir -p $dir/log || exit 1;

for f in $src/final.mdl $src/final.mat $src/splice_opts $src/cmvn_opts; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

cp $src/phones.txt $dir 2>/dev/null

mkdir -p $dir/log

# nnet.config will be a config for a few trivial neural-network layers
# that come before the main network, and which do things like
echo -n >$dir/nnet.config

if [ ! -z "$global_cmvn_stats" ]; then
  [ ! -f $src/cmvn_opts ] && \
    echo "$0: expected $src/cmvn_opts to exist" && exit 1;
  norm_vars=false
  if grep 'norm-means=false' $src/cmvn_opts; then
    echo "$0: if --norm-means=false, don't supply the --global-cmvn-stats option to this script"
    exit 1;
  elif grep 'norm-vars=true' $src/cmvn_opts; then
    echo "$0: warning: this script has not been tested with --norm-vars=true in CMVN options"
    norm_vars=true
  fi


  # First add to the config, layers that will do the same transform as cepstral
  # mean and variance normalization using these global stats.  We do this as
  # first an added offset (FixedBiasComonent), then, only if norm-vars=true
  # in the CMVN options, a scaling (FixedScaleComponent).
  
  $cmd $dir/log/copy_feats.log \
    copy-feats --binary=false "$global_cmvn_stats" $dir/global_cmvn_stats.txt || exit 1;
  cat $dir/global_cmvn_stats.txt | \
    perl -e ' $line0 = <STDIN>; $line0 == "[\n" || die "expected first line to be [, got $line0";
    $line1 = <STDIN>; $line2 = <STDIN>; @L1 = split(" ",$line1); @L2 = split(" ",$line2);
    ($bias_out, $scale_out) = @ARGV;
    open(B, ">$bias_out") || die "opening bias-out file $bias_out";
    open(S, ">$scale_out") || die "opening scale-out file $scale_out";
    pop @L2; pop @L2; # remove the " 0 ]"
    $count = pop @L1;  # last element of line 1 is total count.
    ($count > 0.0) || die "Bad count $count";
    $dim = @L1;
    $dim == scalar @L2 || die "Bad dimension of second line of CMVN stats @L2";
    print B "[ ";  print S "[ ";
    for ($x = 0; $x < $dim; $x++) {
      $mean = $L1[$x] / $count;  $var = ($L2[$x] / $count) - ($mean * $mean);
      $bias = -$mean;  print B "$bias ";
      $scale = 1.0 / sqrt($var); $scale > 0 || die "Bad scale $scale";  print S "$scale ";
    }
    print B "]\n";  print S "]\n"; ' $dir/bias.txt $dir/scales.txt || exit 1;
  echo "FixedBiasComponent bias=$dir/bias.txt" >> $dir/nnet.config  
  if $norm_vars; then
    echo "FixedScaleComponent scales=$dir/scales.txt" >> $dir/nnet.config  
  fi
  echo "--norm-means=false --norm-vars=false" >$dir/cmvn_opts || exit 1;
else
  cp $src/cmvn_opts $dir/ || exit 1;
fi

# We need the dimension of the raw features.  We work it out from the LDA matrix dimension.
# get a word-count of the second row of the LDA matrix...  this will be either the
# spliced dim or the spliced dim plus one.
spliced_dim=$(copy-matrix --binary=false $src/final.mat - | head -n 2 | tail -n 1 | wc -w) || exit 1;


splice_opts=$(cat $src/splice_opts) || exit 1;
# Work out how many frames are spliced together by splicing a matrix with one element
# and testing the resulting number of columns.
num_splice=$(echo "foo [ 1.0 ]" | splice-feats $splice_opts ark:- ark:- | feat-to-dim ark:- -)

# We'll separately need the left-context and right-context.
# defaults in the splice-feats code are 4 and 4.
left_context=4
right_context=4
for opt in $(cat $src/splice_opts); do
  if echo $opt | grep left-context  >/dev/null; then
    left_context=$(echo $opt | cut -d= -f2) || exit 1;
  fi
  if echo $opt | grep right-context  >/dev/null; then
    right_context=$(echo $opt | cut -d= -f2) || exit 1;
  fi
done
if ! [ $num_splice -eq $[$left_context+1+$right_context] ]; then
  echo "$0: num-splice worked out from the binaries differs from our interpreation of the options:"
  echo "$num_splice != $left_context + 1 + $right_context"
  exit 1;
fi

modulo=$[$spliced_dim%$num_splice]
if [ $modulo -eq 1 ]; then
  # matrix includes offset term.
  spliced_dim=$[$spliced_dim-1];
  cp $src/final.mat $dir/
elif [ $modulo -eq 0 ]; then
  # We need to add a zero bias term to the matrix, because the AffineComponent
  # expects that.
  copy-matrix --binary=false $src/final.mat - | \
    awk '{if ($NF == "]") { $NF = "0"; print $0, "]"; } else { if (NF > 1) { print $0, "0"; } else {print;}}}' >$dir/final.mat
else
  echo "$0: Cannot make sense of spliced dimension $spliced_dim and num-splice=$num_splice"
  exit 1;
fi
feat_dim=$[$spliced_dim/$num_splice];
echo "SpliceComponent input-dim=$feat_dim left-context=$left_context right-context=$right_context" >>$dir/nnet.config

# use AffineComponentPreconditioned as it's easier to configure than AffineComponentPreconditionedOnline.
echo "AffineComponentPreconditioned alpha=4.0 learning-rate=$learning_rate max-change=$max_change matrix=$dir/final.mat" >>$dir/nnet.config


$cmd $dir/log/nnet_init.log \
  nnet-init $dir/nnet.config $dir/lda.nnet || exit 1;

$cmd $dir/log/nnet_insert.log \
  nnet-insert --insert-at=0 --randomize-next-component=false \
   $src/final.mdl $dir/lda.nnet $dir/final.mdl || exit 1;

if $cleanup; then
  rm $dir/final.mat $dir/lda.nnet
fi
