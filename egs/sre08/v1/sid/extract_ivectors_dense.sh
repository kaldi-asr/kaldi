#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  David Snyder
#               2016  Matthew Maciejewski
# Apache 2.0.

# This script extracts a set of iVectors over a sliding window
# for a set of utterances, given features and a trained
# iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=-2
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
chunk_size=100
period=50
frame_shift=0.01
frame_length=0.025
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <extractor-dir> <data> <segments-file> <ivector-dir>"
  echo " e.g.: $0 exp/extractor_2048_male data/train_male exp/ivectors_male"
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
data=$2
seg_fi=$3
dir=$4

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
frame_shift_ms=$(awk "BEGIN{print(1000*$frame_shift)}") # Frame shift in milliseconds
frame_length_ms=$(awk "BEGIN{print(1000*$frame_length)}") # Frame length in milliseconds
## Set up features.

# TODO probably should filter $dir/segments by the utt-ids in the split data dir
feats="ark,s:extract-feature-segments --frame-shift=$frame_shift_ms --frame-length=$frame_length_ms scp:$sdata/JOB/feats.scp $dir/segments ark:- | add-deltas $delta_opts ark:- ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"

if [ $stage -le -2 ]; then
  $cmd $dir/log/extract_segments.log \
    segmentation-init-from-segments --frame-shift=$frame_shift --label=1 $seg_fi ark:- \| \
    segmentation-split-segments --split-label=1 --overlap-length=$period --max-segment-length=$chunk_size \
    ark:- ark,scp:$dir/segmentation.ark,$dir/segmentation.scp || exit 1;
fi

if [ $stage -le -1 ]; then
  $cmd $dir/log/create_segments.log \
    segmentation-to-segments --frame-overlap=0 --single-speaker=true --frame-shift=$frame_shift scp:$dir/segmentation.scp ark,t:$dir/utt2spk ark,t:$dir/segments || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  dubm="fgmm-global-to-gmm $srcdir/final.ubm -|"

  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $srcdir/final.ubm "$feats" \
       ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract --verbose=2 $srcdir/final.ie "$feats" ark,s,cs:- \
      ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors and segments across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
fi
