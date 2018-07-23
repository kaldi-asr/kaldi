#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#           2013  Daniel Povey

# This trains a full-covariance UBM from an existing (diagonal or full) UBM,
# for a specified number of iterations.  This is for speaker-id systems
# (we use features specialized for that, and vad).

# Begin configuration section.
nj=16
cmd=run.pl
stage=-2
num_gselect=20 # cutoff for Gaussian-selection that we do once at the start.
subsample=5
num_iters=4
min_gaussian_weight=1.0e-04
remove_low_count_gaussians=true # set this to false if you need #gauss to stay fixed.
cleanup=true
apply_cmn=true # If true, apply sliding window cepstral mean normalization
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/train_full_ubm.sh <data> <old-ubm-dir> <new-ubm-dir>"
  echo "Trains a full-covariance UBM starting from an existing diagonal or"
  echo "full-covariance UBM system."
  echo " e.g.: steps/train_full_ubm.sh --num-iters 8 data/train exp/diag_ubm exp/full_ubm"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|16>                                      # number of parallel training jobs"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # initial model (diagonalized if needed)"
  echo "  --subsample <n|5>                                # Take every n'th sample, for efficiency"
  echo "  --num-iters <n|4>                                # Number of iterations of E-M"
  echo "  --min-gaussian-weight <weight|1.0e-05>           # Minimum Gaussian weight (below this,"
  echo "                                                   # we won't update, and will remove Gaussians"
  echo "                                                   # if --remove-low-count-gaussians is true"
  echo "  --remove-low-count-gaussians <true,false|true>   # If true, remove Gaussians below min-weight"
  echo "                                                   # (will only happen on last iteration, in any case"
  echo "  --cleanup <true,false|true>                      # If true, clean up accumulators, intermediate"
  echo "                                                   # models and gselect info"
  exit 1;
  echo " --apply-cmn <true,false|true>                     # if true, apply sliding window cepstral mean"
  echo "                                                   # normalization to features"
fi

data=$1
srcdir=$2
dir=$3

for f in $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;


delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
if [ -f $srcdir/delta_opts ]; then
  cp $srcdir/delta_opts $dir/ 2>/dev/null
fi

## Set up features.
if $apply_cmn; then
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"
else
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"
fi


if [ $stage -le -2 ]; then
  if [ -f $srcdir/final.dubm ]; then # diagonal-covariance in $srcdir
    $cmd $dir/log/convert_diag_to_full \
      gmm-global-to-fgmm $srcdir/final.dubm $dir/0.ubm || exit 1;
  elif [ -f $srcdir/final.ubm ]; then
    cp $srcdir/final.ubm $dir/0.ubm || exit 1;
  else
    echo "$0: in $srcdir, expecting final.ubm or final.dubm to exist"
    exit 1;
  fi
fi

if [ $stage -le -1 ]; then
  echo "$0: doing Gaussian selection (using diagonal form of model; selecting $num_gselect indices)"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect "fgmm-global-to-gmm $dir/0.ubm - |" "$feats" \
    "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi


x=0
while [ $x -lt $num_iters ]; do
  echo "Pass $x"
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      fgmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $dir/$x.ubm "$feats" \
      $dir/$x.JOB.acc || exit 1;

    if [ $[$x+1] -eq $num_iters ];then
      lowcount_opt="--remove-low-count-gaussians=$remove_low_count_gaussians" # as specified by user.
    else
    # On non-final iters, we in any case can't remove low-count Gaussians because it would
    # cause the gselect info to become out of date.
      lowcount_opt="--remove-low-count-gaussians=false"
    fi
    $cmd $dir/log/update.$x.log \
    fgmm-global-est $lowcount_opt --min-gaussian-weight=$min_gaussian_weight --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.*.acc |" \
      $dir/$[$x+1].ubm || exit 1;
    $cleanup && rm $dir/$x.*.acc $dir/$x.ubm
  fi
  x=$[$x+1]
done

$cleanup && rm $dir/gselect.*.gz

rm $dir/final.ubm 2>/dev/null
mv $dir/$x.ubm $dir/final.ubm || exit 1;
