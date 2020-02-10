#!/usr/bin/env bash

# Copyright     2013  Daniel Povey
#          2014-2017  David Snyder
#               2015  Johns Hopkins University (Author: Daniel Garcia-Romero)
#               2015  Johns Hopkins University (Author: Daniel Povey)
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
use_gpu=true
chunk_size=256
nnet_job_opt=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor_2048_male data/train_male exp/ivectors_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  echo "  --nnet-job-opt <option|''>                       # Options for the DNN jobs which add to or"
  echo "                                                   # replace those specified by --cmd"
  exit 1;
fi

srcdir=$1
nnet=$2
data=$3
data_dnn=$4
dir=$5

gpu_opt=""
if $use_gpu; then
  nnet_job_opt="$nnet_job_opt --gpu 1"
  gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are trying to use the GPU but you have not compiled"
    echo "   for CUDA.  If you have GPUs and have nvcc installed, go to src/"
    echo "   and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be slow."
  gpu_opt="--use-gpu=no"
fi

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

sdata_dnn=$data_dnn/split$nj;
utils/split_data.sh $data_dnn $nj || exit 1;

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

nnet_feats="ark,s,cs:apply-cmvn-sliding --center=true scp:$sdata_dnn/JOB/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  for g in $(seq $nj); do
    $cmd $nnet_job_opt $dir/log/extract_ivectors.$g.log \
      nnet-am-compute $gpu_opt --apply-log=true --chunk-size=${chunk_size} \
        $nnet "`echo $nnet_feats | sed s/JOB/$g/g`" ark:- \
        \| select-voiced-frames ark:- scp,s,cs:$sdata/$g/vad.scp ark:- \
        \| logprob-to-post --min-post=$min_post ark:- ark:- \| \
        scale-post ark:- $posterior_scale ark:- \| \
        ivector-extract --verbose=2 $srcdir/final.ie \
        "`echo $feats | sed s/JOB/$g/g`" ark,s,cs:- \
        ark,scp,t:$dir/ivector.$g.ark,$dir/ivector.$g.scp || exit 1 &
  done
  wait
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector.$j.scp; done >$dir/ivector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Be careful here: the speaker-level iVectors are now length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of iVectors for each speaker and length-normalizing"
  $cmd $dir/log/speaker_mean.log \
    ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
    ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
    ivector-normalize-length ark:- ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector.scp || exit 1;
fi
