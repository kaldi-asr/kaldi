#!/usr/bin/env bash
# Copyright 2015-2017   David Snyder
#           2015        Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015        Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script derives a full-covariance UBM from DNN posteriors and
# speaker recognition features.

# Begin configuration section.
nj=8
cmd="run.pl"
stage=-2
delta_window=3
delta_order=2
use_gpu=true
nnet_job_opt=
cleanup=true
chunk_size=256
stage=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: steps/init_full_ubm_from_dnn.sh <data-speaker-id> <data-dnn> <dnn-model> <new-ubm-dir>"
  echo "Initializes a full-covariance UBM from DNN posteriors and speaker recognition features."
  echo " e.g.: steps/init_full_ubm_from_dnn.sh data/train data/train_dnn exp/dnn/final.mdl exp/full_ubm"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|16>                                      # number of parallel training jobs"
  echo "  --delta-window <n|3>                             # delta window size"
  echo "  --delta-order <n|2>                              # delta order"
  echo "  --use-gpu <true/false>                           # Use GPU to extract DNN posteriors"
  echo "  --chunk-size <n|256>                             # Number of frames processed at a time by the DNN"
  echo "  --nnet-job-opt <option|''>                       # Options for the DNN jobs which add to or"
  echo "                                                   # replace those specified by --cmd"
  exit 1;
fi

data=$1     # Features for the GMM
data_dnn=$2 # Features for the DNN
nnet=$3
dir=$4

gpu_opt=""
nnet_job_opt=""
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


for f in $data/feats.scp $data/vad.scp ${data_dnn}/feats.scp \
    ${data_dnn}/vad.scp $nnet; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

sdata_dnn=$data_dnn/split$nj;
utils/split_data.sh $data_dnn $nj || exit 1;

delta_opts="--delta-window=$delta_window --delta-order=$delta_order"
echo $delta_opts > $dir/delta_opts

logdir=$dir/log

nnet_feats="ark,s,cs:apply-cmvn-sliding --center=true scp:$sdata_dnn/JOB/feats.scp ark:- |"

feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | \
apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | \
select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

# Parse the output of nnet-am-info to find the size of the output layer
# of the TDNN.  This will also correspond to the number of components
# in the ancillary GMM.
num_components=`grep -oP 'output-dim\ \K[0-9]+' <(nnet-am-info $nnet 2> /dev/null)`

if [ $stage -le 0 ]; then
  echo "$0: accumulating stats from DNN posteriors and speaker ID features"
  for g in $(seq $nj); do
    $cmd $nnet_job_opt $dir/log/make_stats.$g.log \
    nnet-am-compute $gpu_opt \
      --chunk-size=${chunk_size} --apply-log=true $nnet \
      "`echo $nnet_feats | sed s/JOB/$g/g`" \
      ark:- \
      \| select-voiced-frames ark:- scp,s,cs:$sdata/$g/vad.scp ark:- \
      \| logprob-to-post ark:- ark:- \| \
      fgmm-global-acc-stats-post ark:- $num_components \
      "`echo $feats | sed s/JOB/$g/g`" \
      $dir/stats.$g.acc || exit 1 &
  done
  wait
fi

if [ $stage -le 1 ]; then
  echo "$0: initializing GMM from stats"
  $cmd $dir/log/init.log \
    fgmm-global-init-from-accs --verbose=2 \
    "fgmm-global-sum-accs - $dir/stats.*.acc |" $num_components \
    $dir/final.ubm || exit 1;
fi

if $cleanup; then
  echo "$0: removing stats"
  for g in $(seq $nj); do
    rm $dir/stats.$g.acc || exit 1
  done
fi
