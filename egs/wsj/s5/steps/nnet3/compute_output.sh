#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#                2016  Vimal Manohar
# Apache 2.0.

# This script does forward propagation through a neural network.

# Begin configuration section.
stage=1
nj=4 # number of jobs.
cmd=run.pl
use_gpu=false
frames_per_chunk=50
iter=final
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
frame_subsampling_factor=1
compress=false    # Specifies whether the output should be compressed before
                  # dumping to disk
online_ivector_dir=
output_name=      # Dump outputs for this output-node
apply_exp=false  # Apply exp i.e. write likelihoods instead of log-likelihoods
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <nnet-dir> <output-dir>"
  echo "e.g.:   steps/nnet3/compute_output.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet3/ivectors_test_eval92 \\"
  echo "    data/test_eval92_hires exp/nnet3/tdnn exp/nnet3/tdnn/output"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

mkdir -p $dir/log

# convert $dir to absolute pathname
fdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

model=$srcdir/$iter.raw
if [ ! -f $srcdir/$iter.raw ]; then
  echo "$0: WARNING: no such file $srcdir/$iter.raw. Trying $srcdir/$iter.mdl instead."
  model=$srcdir/$iter.mdl
fi

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ ! -z "$output_name" ] && [ "$output_name" != "output" ]; then
  echo "$0: Using output-name $output_name"
  model="nnet3-copy --edits='remove-output-nodes name=output;rename-node old-name=$output_name new-name=output' $model - |"
fi

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

## Set up features.
if [ -f $srcdir/final.mat ]; then
  echo "$0: ERROR: lda feature type is no longer supported." && exit 1
fi
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

frame_subsampling_opt=
if [ $frame_subsampling_factor -ne 1 ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
fi

if $apply_exp; then
  output_wspecifier="ark:| copy-matrix --apply-exp ark:- ark,scp:$dir/output.JOB.ark,$dir/output.JOB.scp"
else
  output_wspecifier="ark:| copy-feats --compress=$compress ark:- ark,scp:$dir/output.JOB.ark,$dir/output.JOB.scp"
fi

gpu_opt="--use-gpu=no"
gpu_queue_opt=

if $use_gpu; then
  gpu_queue_opt="--gpu 1"
  suffix="-batch"
  gpu_opt="--use-gpu=yes"
else
  gpu_opt="--use-gpu=no"
fi

if [ $stage -le 2 ]; then
  $cmd $gpu_queue_opt JOB=1:$nj $dir/log/compute_output.JOB.log \
    nnet3-compute$suffix $gpu_opt $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     "$model" "$feats" "$output_wspecifier" || exit 1;
fi

for n in $(seq $nj); do
  cat $dir/output.$n.scp
done > $dir/output.scp

exit 0;
