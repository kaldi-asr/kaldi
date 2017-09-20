#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.

# Begin configuration section.
nj=30
cmd="run.pl"
chunk_size=-1 # If the chunk size over which the embedding is extracted.
              # For example, 6000 means we extract an embedding every 6000
              # frames and average to get a single embedding.  By default,
              # this is -1, which means that we use the entire utterance
              # to extract the embedding.
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.raw $data/feats.scp $data/vad.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

nnet=$srcdir/final.raw
if [ -f $srcdir/extract.config ] ; then
  echo "$0: using $srcdir/extract.config to extract xvectors"
  nnet="nnet3-copy --nnet-config=$srcdir/extract.config $srcdir/final.raw - |"
fi

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting xvectors for $data"
sdata=$data/split$nj/JOB

# Set up the features
feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    nnet3-xvector-compute --chunk-size=$chunk_size --use-gpu=no \
    "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "$0: computing mean of xvectors for each speaker"
  $cmd $dir/log/speaker_mean.log \
    ivector-mean ark:$data/spk2utt scp:$dir/xvector.scp \
    ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
fi
