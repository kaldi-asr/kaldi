#!/bin/bash

# Copyright       2014  Daniel Povey
# Apache 2.0

#
# This script takes a data directory and a directory computed by
# ./train_lvtln_model.sh, and it computes speaker warp-factors spk2warp.  It
# expects vad.scp to exist in the data directory.  Note: like
# train_lvtln_model.sh, it uses features of the speaker-id type, i.e. double
# delta features with sliding window cepstral mean normalization.

# Begin configuration.
stage=0
config=
cmd=run.pl
logdet_scale=0.0
subsample=5 # We use every 5th frame by default; this is more
            # CPU-efficient.
nj=4
cleanup=true
num_gselect=15
refine_transforms=true  # if true, do a second pass of transform estimation.
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-dir> <vtln-dir> <exp-dir>"
   echo "e.g.: $0 data/train_novtln exp/vtln exp/train_warps"
   echo "where <vtln-dir> is produced by train_lvtln_model.sh"
   echo "Output is <exp-dir>/spk2warp"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --nj <num-jobs>                                  # number of jobs to use (default 4)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi

data=$1
vtlndir=$2
dir=$3

for f in $data/feats.scp $data/spk2utt $vtlndir/final.lvtln $vtlndir/final.dubm $vtlndir/final.ali_dubm; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done

if [ -f $data/utt2warp ]; then
  echo "$0: source data directory $data appears to already have VTLN.";
  exit 1;
fi

mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;

cmvn_sliding_opts="--norm-vars=false --center=true --cmn-window=300"
# don't change $cmvn_sliding_opts, it should probably match the
# options used in ../sid/train_diag_ubm.sh and ./train_lvtln_model.sh

sifeats="ark,s,cs:add-deltas scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding $cmvn_sliding_opts ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk $dir/trans.0.JOB ark:- ark:- |"


if [ $stage -le 0 ]; then
  echo "$0: computing Gaussian selection info."

  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $vtlndir/final.ali_dubm "$sifeats" \
      "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi
  

if [ $stage -le 0 ]; then
  echo "$0: computing initial LVTLN transforms"

  $cmd JOB=1:$nj $dir/log/lvtln.0.JOB.log \
    gmm-global-gselect-to-post $dir/final.ali_dubm "$sifeats" \
      "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark:- \| \
    gmm-global-est-lvtln-trans --spk2utt=$sdata/JOB/spk2utt \
      --logdet-scale=$logdet_scale --verbose=1 \
      $vtlndir/final.dubm $vtlndir/final.lvtln "$sifeats" ark,s,cs:- \
      ark:$dir/trans.0.JOB ark,t:$dir/warp.0.JOB || exit 1
  
  # consolidate the warps into one file.
  for j in $(seq $nj); do cat $dir/warp.0.$j; done > $dir/warp.0
  rm $dir/warp.0.*
fi

if $refine_transforms; then
  ln -sf warp.0 $dir/spk2warp
  $cleanup && rm $dir/gselect.*.gz $dir/trans.0.*
  echo "$0: --refine-transforms=false so exiting with current warps."
  echo "$0: Distribution of classes for one job is below."
  grep 'Distribution of classes' $dir/log/lvtln.0.1.log
  exit 0;
fi

if [ $stage -le 1 ]; then
  echo "$0: computing refined LVTLN transforms"
  
  $cmd JOB=1:$nj $dir/log/lvtln.1.JOB.log \
    gmm-global-gselect-to-post $dir/final.dubm "$feats" \
      "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark:- \| \
    gmm-global-est-lvtln-trans --spk2utt=$sdata/JOB/spk2utt \
      --logdet-scale=$logdet_scale --verbose=1 \
      $vtlndir/final.dubm $vtlndir/final.lvtln "$sifeats" ark,s,cs:- \
      ark:/dev/null ark,t:$dir/warp.1.JOB || exit 1
  
  # consolidate the warps into one file.
  for j in $(seq $nj); do cat $dir/warp.1.$j; done > $dir/warp.1
  rm $dir/warp.1.*

  ns1=$(cat $dir/0.warp | wc -l)
  ns2=$(cat $dir/1.warp | wc -l)
  ! [ "$ns1" == "$ns2" ] && echo "$0: Number of speakers differ pass1 vs pass2, $ns1 != $ns2" && exit 1;
  paste $dir/0.warp $dir/1.warp | awk '{x=$2 - $4; if ((x>0?x:-x) > 0.010001) { print $1, $2, $4; }}' > $dir/warp_changed
  nc=$(cat $dir/warp_changed | wc -l)
  echo "$0: For $nc speakers out of $ns1, warp changed pass1 vs pass2 by >0.01, see $dir/warp_changed for details"
fi

$cleanup && rm $dir/gselect.*.gz $dir/trans.0.*

ln -sf warp.1 $dir/spk2warp

echo "$0: created warp factors in $dir/spk2warp"

echo "$0: Distribution of classes for one job is below."
grep 'Distribution of classes' $dir/log/lvtln.1.1.log

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

echo "$0: Done training LVTLN model in $dir"
