#!/bin/bash
# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

# This script gets frame and utt level LRE matrix scores from a nnet3.


stage=0
nj=8 # number of parallel jobs
cmd=run.pl

vad=false

# typical options for nnet3-compute
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [options] <nnet_dir> <feat_dir> <lang_ali> <score_dir>"
  exit 1;
fi


mdl=$1/final.raw
data=$2
ali_dir=$3
dir=$4
mkdir -p $dir/log

sdata=$data/split$nj 
utils/split_data.sh $data $nj


if [ $stage -le 0 ]; then
  echo "$0: get outputs of nnet3 in both frame and utt-level"
  if $vad; then
    $cmd JOB=1:$nj $dir/log/nnet3-compute.JOB.log \
      select-voiced-frames scp:$sdata/JOB/feats.scp scp,s,cs:$sdata/JOB/vad.scp ark:- \| \
      nnet3-compute --use-gpu=no --frames-per-chunk=$frames_per_chunk \
        --extra-left-context=$extra_left_context --extra-right-context=$extra_right_context \
        --extra-left-context-initial=$extra_left_context_initial --extra-right-context-final=$extra_right_context_final \
      $mdl ark:- ark,t:$dir/raw_output.JOB.ark
  else
    $cmd JOB=1:$nj $dir/log/nnet3-compute.JOB.log \
      nnet3-compute --use-gpu=no --frames-per-chunk=$frames_per_chunk \
        --extra-left-context=$extra_left_context --extra-right-context=$extra_right_context \
        --extra-left-context-initial=$extra_left_context_initial --extra-right-context-final=$extra_right_context_final \
      $mdl scp:$sdata/JOB/feats.scp ark,t:$dir/raw_output.JOB.ark
  fi

  if [ -f $dir/output.ark.frame ]; then
    rm $dir/output.ark.frame
  fi
  for job in `seq $nj`; do
    cat $dir/raw_output.$job.ark >> $dir/output.ark.frame
    # rm $dir/raw_output.$job.ark
  done
  matrix-sum-rows-mean ark:$dir/output.ark.frame ark,t:$dir/output.ark.utt
fi

if [ $stage -le 1 ]; then
  echo "$0: convert frame and utt-level output to LRE matrix scores"
  lang_names=`cat $ali_dir/lang2lang_id | awk '{print $1}' | sed ':a;N;$!ba;s/\n/ /g'`
  lang_names="\ \ \ \ "$lang_names
  sed -i 's/\[//g;s/\]//g' $dir/output.ark.frame $dir/output.ark.utt
  python local/nnet3/output_format_frame.py $dir/output.ark.frame $dir/output.ark.frame_
  mv $dir/output.ark.frame_ $dir/output.ark.frame
  sed -i "1i$lang_names" $dir/output.ark.frame $dir/output.ark.utt
  echo "Frame and utt level matrix scores prepared."
fi


