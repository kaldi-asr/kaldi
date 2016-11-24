#!/bin/bash

# Copyright     2013  Daniel Povey
#          2014-2015  David Snyder
#               2015  Johns Hopkins University (Author: Daniel Garcia-Romero)
#               2015  Johns Hopkins University (Author: Daniel Povey)
#               2016  Go-Vivace Inc. (Author: Mousmita Sarma) 
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained DNN-based iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successive features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0  <extractor-dir> <dnn-model> <data-language-id> <data-dnn> <ivectors-dir>"
  echo " e.g.: $0 exp/extractor_dnn exp/nnet2_online/nnet_ms_a/final.mdl data/lre07 data/lre07_dnn exp/ivectors_lre07"  
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  exit 1;
fi

srcdir=$1
nnet=$2
data=$3
data_dnn=$4
dir=$5

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

sdata_dnn=$data_dnn/split$nj;
utils/split_data.sh $data_dnn $nj || exit 1;
          

# Set up features.
feats="ark,s,cs:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$sdata/JOB/feats.scp ark:- | add-deltas-sdc ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

nnet_feats="ark,s,cs:apply-cmvn-sliding --center=true scp:$sdata_dnn/JOB/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    nnet-am-compute --apply-log=true $nnet "$nnet_feats" ark:- \
    \| select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- \
    \| logprob-to-post --min-post=$min_post ark:- ark:- \| \
    scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract --verbose=2 $srcdir/final.ie "$feats" ark,s,cs:- \
      ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
fi

