#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This computes an online alignment model (final.online_alimdl) for use in online decoding.
# This is done with apply-cmvn-sliding.


# Begin configuration section.
stage=0
cmd=run.pl
cmvn_sliding_config=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data> <exp-dir>"
  echo " e.g.: $0 --cmvn-sliding-config conf/sliding.conf data/train exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --cmvn-sliding-config <cmvn-config-file>         # config file containing options for"
  echo "                                                   # apply-cmvn-sliding"
  echo "  --config <config-file>                           # config containing options for this script"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

data=$1
dir=$2

for f in $data/feats.scp $dir/final.mdl $dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $dir/num_jobs` || exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
norm_vars=`cat $dir/norm_vars 2>/dev/null` || norm_vars=false # cmn/cmvn option, default false.

mkdir -p $dir/log # should already exist..


if $norm_vars; then
  echo "$0: warning: the default features are normalizing variances; make sure"
  echo "    your apply-cmvn-sliding config does the same, if that is what you want."
  echo "    (this setup may not be fully tested for this case)"
fi

# Set up features.

if [ -f $dir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

if [ ! -z "$cmvn_sliding_config" ]; then
  cmvn_sliding_config_opt="--config=$cmvn_sliding_config"
else
  cmvn_sliding_config_opt=
fi

## Set up speaker-independent features.
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";
         onlinefeats="ark,s,cs:apply-cmvn-sliding $cmvn_sliding_config_opt scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    onlinefeats="ark,s,cs:apply-cmvn-sliding $cmvn_sliding_config_opt scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $dir/trans.1 ]; then
  echo "$0: Using fMLLR transforms from $dir"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$dir
else 
  feats="$sifeats"
fi

if [ $stage -le 0 ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
  $cmd JOB=1:$nj $dir/log/acc_online_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/final.mdl "$feats" "$onlinefeats" \
    ark,s,cs:- $dir/final.JOB.acc || exit 1;
  [ `ls $dir/final.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_online_alimdl.log \
    gmm-est --remove-low-count-gaussians=false $dir/final.mdl \
    "gmm-sum-accs - $dir/final.*.acc|" $dir/final.online_alimdl  || exit 1;
  rm $dir/final.*.acc
fi


echo Done
