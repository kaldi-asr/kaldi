#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  David Snyder
# Apache 2.0.

# TODO This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.  It is based on the
# script sid/extract_ivectors.sh, but uses ivector-extract-dense
# to extract ivectors from overlapping chunks.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
chunk_size=150
period=50
min_chunk_size=25
use_vad=false
pca_dim=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor data/train exp/ivectors"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  echo "  --chunk-size <n|150>                             # Size of chunks from which to extract iVectors"
  echo "  --period <n|50>                                  # Frequency that iVectors are computed"
  echo "  --min-chunks-size <n|25>                         # Minimum chunk-size after splitting larger segments"
  echo "  --use-vad <bool|false>                           # If true, use vad.scp instead of segments"
  exit 1;
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

if $use_vad ; then
  [ ! -f $data/vad.scp ] && echo "No such file $data/vad.scp" && exit 1;
else
  [ ! -f $data/segments ] && echo "No such file $data/segments" && exit 1;
fi
# Set various variables.

mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
echo $nj > $dir/num_jobs

if $use_vad ; then
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
else
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- |"
fi

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  dubm="fgmm-global-to-gmm $srcdir/final.ubm -|"

  if $use_vad ; then
    $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
      fgmm-global-gselect-to-post --min-post=$min_post $srcdir/final.ubm "$feats" \
         ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
      ivector-extract-dense --verbose=2 --chunk-size=$chunk_size \
        --min-chunk-size=$min_chunk_size --period=$period $srcdir/final.ie \
        "$feats" ark,s,cs:- \
        ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp \
        ark,t:$dir/utt2spk.JOB || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
      fgmm-global-gselect-to-post --min-post=$min_post $srcdir/final.ubm "$feats" \
         ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
      ivector-extract-dense --verbose=2 --chunk-size=$chunk_size \
        --min-chunk-size=$min_chunk_size --period=$period $srcdir/final.ie \
        "$feats" ark,s,cs:- $sdata/JOB/segments \
        ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp \
        $dir/segments.JOB ark,t:$dir/utt2spk.JOB || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
  for j in $(seq $nj); do cat $dir/utt2spk.$j; done >$dir/utt2spk || exit 1;
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt || exit 1;
  if ! $use_vad ; then
    for j in $(seq $nj); do cat $dir/segments.$j; done >$dir/segments || exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: Computing mean of iVectors"
  $cmd $dir/log/mean.log \
    ivector-mean scp:$dir/ivector.scp $dir/mean.vec || exit 1;
fi

if [ $stage -le 3 ]; then
  if [ -z "$pca_dim" ]; then
    pca_dim=-1
  fi
  echo "$0: Computing whitening transform"
  $cmd $dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$dir/ivector.scp $dir/transform.mat || exit 1;
fi
