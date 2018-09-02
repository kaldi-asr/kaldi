#!/bin/bash
# Copyright  2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Danwei Cai)
#            2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Ming Li) 
# Apache 2.0.
#
# Extract tandem features to replace MFCC features in conventional i-vector/PLDA
# pipeline.
# DNN acoustic model is used to extract the frame level phoneme posterior probabilities.
# After log, PCA, the resulted low dimensional features are fused with MFCC at the 
# feature level to get hybrid tandem feature.

# Begin configuration section.
use_gpu=yes
nj=4
cmd="run.pl"
delta_window=3
delta_order=2
chunk_size=512
# End configuration section.
delta_opts="--delta-window=$delta_window --delta-order=$delta_order"

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

nnet=$1
pca_tmatrix=$2
mfcc_dir=$3
mfcc_hires_dir=$4
dir=$5
log_dir=$dir/log
tandem_dir=$6

# use "name" as part of name of the archive.
name=`basename $mfcc_dir`

mkdir -p $dir || exit 1;
mkdir -p $log_dir || exit 1
hsdata=$mfcc_hires_dir/split$nj;
sdata=$mfcc_dir/split$nj;
utils/split_data.sh $mfcc_hires_dir $nj || exit 1;
utils/split_data.sh $mfcc_dir $nj || exit 1;

mfcc_feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
pppp_feats="ark,s,cs:apply-cmvn-sliding --center=true scp,s,cs:$hsdata/JOB/feats.scp ark:- | nnet-am-compute --apply-log=true --use-gpu=$use_gpu --chunk-size=${chunk_size} $nnet ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$hsdata/JOB/vad.scp ark:- | transform-feats $pca_tmatrix ark:- ark:- |"

$cmd JOB=1:$nj $log_dir/$name.JOB.log \
  paste-feats "$mfcc_feats" "$pppp_feats" \
  ark,scp:$dir/tandem_${name}.JOB.ark,$dir/tandem_${name}.JOB.scp || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $dir/tandem_${name}.$n.scp || exit 1;
done > $tandem_dir/feats.scp || exit 1;

utils/fix_data_dir.sh $tandem_dir || exit 1;
exit 0;

