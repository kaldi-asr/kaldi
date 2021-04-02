#!/usr/bin/env bash

# Copyright       2014  Daniel Povey
# Apache 2.0

#
# This training script computes some things you will need in order to
# extract VTLN-warped features.  It takes as input the data directory
# and an already-trained diagonal-covariance UBM.  Note: although this
# script is in the lid/ directory, because it is intended to be
# used in language identification, but it uses features of the
# same type as those used in the speaker-id scripts (see ../sid/),
# i.e. double-delta features, rather than the "shifted delta cepstra"
# features commonly used in language id.
#
# This script works with either mfcc or plp features; for plp features, you will
# need to set the --base-feat-type option.  Regardless, you will need to set the
# --mfcc-config or --plp-config option if your feature-extraction config is not
# called conf/${base_feat_type}.conf.  The output of this script will be in
# $dir/final.lvtln and $dir/final.dubm and $dir/final.ali_dubm; the directory
# can be passed to ./get_vtln_warps.sh to get VTLN warps for a data directory,
# or (for data passed to this script) you can use the warping factors this
# script outputs in $dir/final.warp
#

# Begin configuration.
stage=-4 #  This allows restarting after partway, when something when wrong.
config=
cmd=run.pl
num_iters=15    # Number of iterations of training.
num_utt_lvtln_init=400 # number of utterances (subset) to initialize
                        # LVTLN transform.  Not too critical.
min_warp=0.85
max_warp=1.25
warp_step=0.01
base_feat_type=mfcc # or could be PLP.
mfcc_config=conf/mfcc.conf  # default, can be overridden.
plp_config=conf/plp.conf  # default, can be overridden.
logdet_scale=0.0
subsample=5 # We use every 5th frame by default; this is more
            # CPU-efficient.
min_gaussian_weight=0.0001 # does not matter; inherited from diag-ubm training script.
nj=4
cleanup=true
num_gselect=15
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

num_classes=$(perl -e "print int(1.5 + ($max_warp - $min_warp) / $warp_step);") || exit 1;
default_class=$(perl -e "print int(0.5 + (1.0 - $min_warp) / $warp_step);") || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-dir> <diag-ubm-dir> <exp-dir>"
   echo "e.g.: $0 data/train_vtln exp/diag_ubm_vtln exp/vtln"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --nj <num-jobs>                                  # number of jobs to use (default 4)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   echo "  --num-iters <num-iters>                          # number of iterations of training"
   echo "  --base-feat-type <feat-type>                     # mfcc or plp, mfcc is default"
   echo "  --mfcc-config <config>                           # config for MFCC extraction, default is"
   echo "                                                   # conf/mfcc.conf"
   echo "  --plp-config <config>                            # config for PLP extraction, default is"
   echo "                                                   # conf/plp.conf"
   exit 1;
fi

data=$1
ubmdir=$2
dir=$3

for f in $data/feats.scp $ubmdir/final.dubm; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done


mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;

cmvn_sliding_opts="--norm-vars=false --center=true --cmn-window=300"
# don't change $cmvn_sliding_opts, it should probably match the
# options used in ../sid/train_diag_ubm.sh.
sifeats="ark,s,cs:add-deltas scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding $cmvn_sliding_opts ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"


# for the subsets of features that we use to estimate the linear transforms, we
# don't bother with CMN.  This will give us wrong offsets on the transforms,
# but it won't matter because we will allow an arbitrary bias term when we apply
# these transforms.

# you need to define CLASS when invoking $cmd on featsub_warped.
featsub_warped="ark:add-deltas ark:$dir/feats.CLASS.ark ark:- | select-voiced-frames ark:- scp,s,cs:$data/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"
featsub_unwarped="ark:add-deltas ark:$dir/feats.$default_class.ark ark:- | select-voiced-frames ark:- scp,s,cs:$data/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"


if [ -f $data/utt2warp ]; then
  echo "$0: source data directory $data appears to already have VTLN.";
  exit 1;
fi

# create a small subset of utterances for purposes of initializing the LVTLN transform
# utils/shuffle_list.pl is deterministic, unlike sort -R.
cat $data/utt2spk | awk '{print $1}' | utils/shuffle_list.pl | \
  head -n $num_utt_lvtln_init > $dir/utt_subset

if [ $stage -le -4 ]; then
  echo "$0: computing warped subset of features"
  if [ -f $data/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    subset_feats="utils/filter_scp.pl $dir/utt_subset $data/segments | extract-segments scp:$data/wav.scp - ark:- "
  else
    echo "$0 [info]: no segments file exists: using wav.scp directly."
    subset_feats="utils/filter_scp.pl $dir/utt_subset $data/wav.scp | wav-copy scp:- ark:- "
  fi
  rm $dir/.error 2>/dev/null
  for c in $(seq 0 $[$num_classes-1]); do
    this_warp=$(perl -e "print ($min_warp + ($c*$warp_step));")
    this_config=""
    if [ "$base_feat_type" = "mfcc" ]; then
      this_config="$mfcc_config"
    else
      this_config="$plp_config"
    fi 
    $cmd $dir/log/compute_warped_feats.$c.log \
      $subset_feats \| compute-${base_feat_type}-feats --verbose=2 \
      --config=$this_config --vtln-warp=$this_warp ark:- ark:- \| \
      copy-feats --compress=true ark:- ark:$dir/feats.$c.ark || touch $dir/.error &
  done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: Computing warped features failed: check $dir/log/compute_warped_feats.*.log"
    exit 1;
  fi
fi

if ! utils/filter_scp.pl $dir/utt_subset $data/feats.scp | \
  compare-feats --threshold=0.95 scp:-  ark:$dir/feats.$default_class.ark >&/dev/null; then
  echo "$0: features stored on disk differ from those computed with no warping."
  echo "    Possibly your feature type is wrong (--base-feat-type option)"
  exit 1;
fi
  
if [ -f $data/segments ]; then
  subset_utts="ark:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- |"
else
  echo "$0 [info]: no segments file exists: using wav.scp directly."
  subset_utts="ark:wav-copy scp:$sdata/JOB/wav.scp ark:- |"
fi

if [ $stage -le -3 ]; then
  echo "$0: initializing base LVTLN transforms in $dir/0.lvtln (ignore warnings below)"
  dim=$(feat-to-dim "$featsub_unwarped" - ) || exit 1;

  $cmd $dir/log/init_lvtln.log \
    gmm-init-lvtln --dim=$dim --num-classes=$num_classes --default-class=$default_class \
      $dir/0.lvtln || exit 1;

  for c in $(seq 0 $[$num_classes-1]); do
    this_warp=$(perl -e "print ($min_warp + ($c*$warp_step));")
    orig_feats=ark:$dir/feats.$default_class.ark
    warped_feats=ark:$dir/feats.$c.ark
    logfile=$dir/log/train_special.$c.log
    this_featsub_warped="$(echo $featsub_warped | sed s/CLASS/$c/)"
    if ! gmm-train-lvtln-special --warp=$this_warp --normalize-var=true \
      $c $dir/0.lvtln $dir/0.lvtln \
      "$featsub_unwarped" "$this_featsub_warped" 2>$logfile; then
      echo "$0: Error training LVTLN transform, see $logfile";
      exit 1;
    fi
  done  
fi

cp $ubmdir/final.dubm $dir/0.dubm

if [ $stage -le -2 ]; then
  echo "$0: computing Gaussian selection info."

  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $ubmdir/final.dubm "$sifeats" \
      "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi
  

if [ $stage -le -1 ]; then
  echo "$0: computing initial LVTLN transforms"  # do this per-utt.

  $cmd JOB=1:$nj $dir/log/lvtln.0.JOB.log \
    gmm-global-gselect-to-post $dir/0.dubm "$sifeats" \
      "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark:- \| \
    gmm-global-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 \
      $dir/0.dubm $dir/0.lvtln "$sifeats" ark,s,cs:- ark:$dir/trans.0.JOB ark,t:$dir/warp.0.JOB || exit 1
  
  # consolidate the warps into one file.
  for j in $(seq $nj); do cat $dir/warp.0.$j; done > $dir/warp.0
  rm $dir/warp.0.*
fi


x=0
while [ $x -lt $num_iters ]; do
  feats="$sifeats transform-feats ark:$dir/trans.$x.JOB ark:- ark:- |"

  # First update the model.
  if [ $stage -le $x ]; then
    echo "$0: Updating model on pass $x"
    # Accumulate stats.
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" \
       $dir/$x.dubm "$feats" $dir/$x.JOB.acc || exit 1;

    $cmd $dir/log/update.$x.log \
      gmm-global-est --remove-low-count-gaussians=false --min-gaussian-weight=$min_gaussian_weight \
        $dir/$x.dubm "gmm-global-sum-accs - $dir/$x.*.acc|" \
      $dir/$[$x+1].dubm || exit 1;
    $cleanup && rm $dir/$x.*.acc $dir/$x.dubm
  fi

  # Now update the LVTLN transforms (and warps.)
  if [ $stage -le $x ]; then
    echo "$0: re-estimating LVTLN transforms on pass $x"
    $cmd JOB=1:$nj $dir/log/lvtln.$x.JOB.log \
      gmm-global-gselect-to-post $dir/$[$x+1].dubm "$feats" \
        "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark:- \| \
      gmm-global-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 \
        $dir/$[$x+1].dubm $dir/0.lvtln "$sifeats" ark,s,cs:- \
        ark:$dir/trans.$[$x+1].JOB ark,t:$dir/warp.$[$x+1].JOB || exit 1

    # consolidate the warps into one file.
    for j in $(seq $nj); do cat $dir/warp.$[$x+1].$j; done > $dir/warp.$[$x+1]
    rm $dir/warp.$[$x+1].*
    $cleanup && rm $dir/trans.$x.*
  fi
  x=$[$x+1]
done

feats="$sifeats transform-feats ark:$dir/trans.$x.JOB ark:- ark:- |"

if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is computed with the
  # speaker-independent features, but matches Gaussian-for-Gaussian with the
  # final speaker-adapted model.
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    gmm-global-acc-stats-twofeats "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" \
      $dir/$x.dubm "$feats" "$sifeats" $dir/$x.JOB.acc || exit 1
  [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-global-est --min-gaussian-weight=$min_gaussian_weight \
      --remove-low-count-gaussians=false $dir/$x.dubm \
     "gmm-global-sum-accs - $dir/$x.*.acc|" $dir/$x.ali_dubm  || exit 1;
  $cleanup && rm $dir/$x.*.acc
fi

if true; then # Diagnostics
  ln -sf warp.$x $dir/final.warp
  if [ -f $data/spk2gender ]; then 
    # To make it easier to eyeball the male and female speakers' warps
    # separately, separate them out.
    for g in m f; do # means: for gender in male female
      cat $dir/final.warp | \
        utils/filter_scp.pl <(grep -w $g $data/spk2gender | awk '{print $1}') > $dir/final.warp.$g
      echo -n "The last few warp factors for gender $g are: "
      tail -n 10 $dir/final.warp.$g | awk '{printf("%s ", $2);}'; 
      echo
    done
  fi
fi

ln -sf $x.dubm $dir/final.dubm
ln -sf $x.ali_dubm $dir/final.ali_dubm
ln -sf 0.lvtln $dir/final.lvtln

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

echo "$0: Done training LVTLN model in $dir"
