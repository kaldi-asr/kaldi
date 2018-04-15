#!/bin/bash
# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

# This script extracts d-vectors with a dnn like tdnn or lstm.


stage=0
nj=8 # number of parallel jobs
cmd=run.pl

# typical options for nnet3-compute
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <nnet3-dir> <data-dir> <dvector-dir>"
  exit 1;
fi


mdl=$1/final.raw.last_hid_out
data=$2
dir=$3
mkdir -p $dir/log

sdata=$data/split$nj 
utils/split_data.sh $data $nj


if [ $stage -le 0 ]; then
  echo "$0: extracting d-vectors"
  $cmd JOB=1:$nj $dir/log/extract_dvectors.JOB.log \
    select-voiced-frames scp:$sdata/JOB/feats.scp scp,s,cs:$sdata/JOB/vad.scp ark:- \| \
    nnet3-compute --use-gpu=no --frames-per-chunk=$frames_per_chunk \
      --extra-left-context=$extra_left_context --extra-right-context=$extra_right_context \
      --extra-left-context-initial=$extra_left_context_initial --extra-right-context-final=$extra_right_context_final \
    $mdl ark:- ark:- \| \
    matrix-sum-rows-mean ark:- ark,scp:$dir/dvector.JOB.ark,$dir/dvector.JOB.scp 

fi

if [ $stage -le 1 ]; then
  echo "$0: combining d-vectors across jobs"
  for n in $(seq $nj); do
    cat $dir/dvector.$n.scp || exit 1;
  done > $dir/vector.scp
fi

if [ $stage -le 2 ]; then
  # the speaker-level d-vectors are length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of d-vectors for each speaker and length-normalizing"
  $cmd $dir/log/speaker_mean.log \
    ivector-normalize-length scp:$dir/vector.scp  ark:- \| \
    ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
    ivector-normalize-length ark:- ark,scp:$dir/spk_vector.ark,$dir/spk_vector.scp || exit 1;
fi

echo "d-vector extraction done."

