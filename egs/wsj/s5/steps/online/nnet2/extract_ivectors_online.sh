#!/bin/bash

# Copyright     2013  Daniel Povey
# Apache 2.0.

# This script is based on ^/egs/sre08/v1/sid/extract_ivectors.sh.  Instead of
# extracting a single iVector per utterance, it extracts one every few frames
# (controlled by the --ivector-period option, e.g. 10, which is to save compute).
# This is used in training (and not-really-online testing) of neural networks
# for online decoding.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
ivector_period=10
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.
compress=true       # If true, compress the features on disk (lossy compression, as used
                    # for feature files.)

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <extractor-dir> <ivector-dir>"
  echo " e.g.: $0 data/train exp/nnet2_online/extractor exp/nnet2_online/ivectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

for f in $data/feats.scp $srcdir/final.ie $srcdir/final.dubm $srcdir/global_cmvn.stats $srcdir/splice_opts \
     $srcdir/online_cmvn.conf; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done


# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

echo $ivector_period > $dir/ivector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)


## Set up features.  $gmm_feats is the version of the features with online CMVN, that we use
## to get the Gaussian posteriors, $feats is the version of the features with no CMN.
gmm_feats="ark,s,cs:apply-cmvn-online --config=$srcdir/online_cmvn.conf $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
feats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"



if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"

  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm \
      "$gmm_feats" ark:- \| scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract-online --ivector-period=$ivector_period $srcdir/final.ie "$feats" ark,s,cs:- ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp,t:$dir/ivector_online.JOB.ark,$dir/ivector_online.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi
