#!/bin/bash
# Copyright   2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Danwei Cai)
#             2017   Speech and Multimodal Intelligent Information Processing Lab, SYSU (Author: Ming Li)
# Apache 2.0.
#
# Extract Phone Posterior Probability 

# Begin configuration section.
nj=4
cmd="run.pl"
use_gpu=yes
chunk_size=512
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

nnet=$1
data=$2
dir=$3

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

logdir=$dir/log

nnet_feats="ark,s,cs:apply-cmvn-sliding --center=true scp:$sdata/JOB/feats.scp ark:- |"

$cmd JOB=1:$nj $dir/log/make_ppp.JOB.log \
  nnet-am-compute --apply-log=true --use-gpu=$use_gpu --chunk-size=${chunk_size} \
   $nnet "$nnet_feats" ark:-\| select-voiced-frames ark:- scp:$sdata/JOB/vad.scp \
   ark,scp:$dir/ppp.JOB.ark,$dir/ppp.JOB.scp || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $dir/ppp.$n.scp || exit 1;
done > ${data}/ppp.scp || exit 1

exit 0;

