#!/usr/bin/env bash

# Copyright   2012  Johns Hopkins University (Author: Daniel Povey)
#             2013  Daniel Povey
#             2014  David Snyder
# Apache 2.0.

# This is a modified version of steps/train_diag_ubm.sh, specialized for
# speaker-id, that does not require to start with a trained model, that applies
# sliding-window CMVN, and that expects voice activity detection (vad.scp) in
# the data directory.  We initialize the GMM using gmm-global-init-from-feats,
# which sets the means to random data points and then does some iterations of
# E-M in memory.  After the in-memory initialization we train for a few
# iteration in parallel.


# Begin configuration section.
nj=4
cmd=run.pl
num_iters=4
stage=-2
num_gselect=30 # Number of Gaussian-selection indices to use while training
               # the model.
num_frames=500000 # number of frames to keep in memory
num_iters_init=20
initial_gauss_proportion=0.5 # Start with half the target number of Gaussians
subsample=5 # subsample all features with this periodicity, in the main E-M phase.
cleanup=true
min_gaussian_weight=0.0001
remove_low_count_gaussians=true # set this to false if you need #gauss to stay fixed.
num_threads=32
parallel_opts="--num-threads 32"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0  <data> <num-gauss> <output-dir>"
  echo " e.g.: $0 data/train 1024 exp/diag_ubm"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-jobs|4>                                # number of parallel jobs to run."
  echo "  --num-iters <niter|20>                           # number of iterations of parallel "
  echo "                                                   # training (default: $num_iters)"
  echo "  --stage <stage|-2>                               # stage to do partial re-run from."
  echo "  --num-gselect <n|30>                             # Number of Gaussians per frame to"
  echo "                                                   # limit computation to, for speed"
  echo " --subsample <n|5>                                 # In main E-M phase, use every n"
  echo "                                                   # frames (a speedup)"
  echo "  --num-frames <n|500000>                          # Maximum num-frames to keep in memory"
  echo "                                                   # for model initialization"
  echo "  --num-iters-init <n|20>                          # Number of E-M iterations for model"
  echo "                                                   # initialization"
  echo " --initial-gauss-proportion <proportion|0.5>       # Proportion of Gaussians to start with"
  echo "                                                   # in initialization phase (then split)"
  echo " --num-threads <n|32>                              # number of threads to use in initialization"
  echo "                                                   # phase (must match with parallel-opts option)"
  echo " --parallel-opts <string|'--num-threads 32'>             # Option should match number of threads in"
  echo "                                                   # --num-threads option above"
  echo " --min-gaussian-weight <weight|0.0001>             # min Gaussian weight allowed in GMM"
  echo "                                                   # initialization (this relatively high"
  echo "                                                   # value keeps counts fairly even)"
  exit 1;
fi

data=$1
num_gauss=$2
dir=$3

! [ $num_gauss -gt 0 ] && echo "Bad num-gauss $num_gauss" && exit 1;

sdata=$data/split$nj
mkdir -p $dir/log
utils/split_data.sh $data $nj || exit 1;

for f in $data/feats.scp $data/vad.scp; do
   [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1
done


# Note: there is no point subsampling all_feats, because gmm-global-init-from-feats
# effectively does subsampling itself (it keeps a random subset of the features).
all_feats="ark,s,cs:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$data/feats.scp ark:- | add-deltas-sdc ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$data/vad.scp ark:- |"

feats="ark,s,cs:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$sdata/JOB/feats.scp ark:- | add-deltas-sdc ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"

num_gauss_init=$(perl -e "print int($initial_gauss_proportion * $num_gauss); ");
! [ $num_gauss_init -gt 0 ] && echo "Invalid num-gauss-init $num_gauss_init" && exit 1;

if [ $stage -le -2 ]; then
  echo "$0: initializing model from E-M in memory, "
  echo "$0: starting from $num_gauss_init Gaussians, reaching $num_gauss;"
  echo "$0: for $num_iters_init iterations, using at most $num_frames frames of data"

  $cmd $parallel_opts $dir/log/gmm_init.log \
    gmm-global-init-from-feats --num-threads=$num_threads --num-frames=$num_frames \
     --min-gaussian-weight=$min_gaussian_weight \
     --num-gauss=$num_gauss --num-gauss-init=$num_gauss_init --num-iters=$num_iters_init \
    "$all_feats" $dir/0.dubm || exit 1;
fi

# Store Gaussian selection indices on disk-- this speeds up the training passes.
if [ $stage -le -1 ]; then
  echo Getting Gaussian-selection info
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/0.dubm "$feats" \
      "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

echo "$0: will train for $num_iters iterations, in parallel over"
echo "$0: $nj machines, parallelized with '$cmd'"

for x in `seq 0 $[$num_iters-1]`; do
  echo "$0: Training pass $x"
  if [ $stage -le $x ]; then
  # Accumulate stats.
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" \
      $dir/$x.dubm "$feats" $dir/$x.JOB.acc || exit 1;
    if [ $x -lt $[$num_iters-1] ]; then # Don't remove low-count Gaussians till last iter,
      opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
    else
      opt="--remove-low-count-gaussians=$remove_low_count_gaussians"
    fi
    $cmd $dir/log/update.$x.log \
      gmm-global-est $opt --min-gaussian-weight=$min_gaussian_weight $dir/$x.dubm "gmm-global-sum-accs - $dir/$x.*.acc|" \
      $dir/$[$x+1].dubm || exit 1;
    $cleanup && rm $dir/$x.*.acc $dir/$x.dubm
  fi
done

$cleanup && rm $dir/gselect.*.gz

mv $dir/$num_iters.dubm $dir/final.dubm || exit 1;
exit 0;
