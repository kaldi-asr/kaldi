#!/bin/bash

# Copyright       2014  Daniel Povey
# Apache 2.0

#
# This script takes a data directory and a directory computed by
# ./train_lvtln_model.sh, and it computes per-utterance warp-factors utt2warp.  It
# expects vad.scp to exist in the data directory.  Note: like
# train_lvtln_model.sh, it uses features of the speaker-id type, i.e. double
# delta features with sliding window cepstral mean normalization.

# Begin configuration.
stage=-1
config=
cmd=run.pl
logdet_scale=0.0
subsample=5 # We use every 5th frame by default; this is more
            # CPU-efficient.
nj=4
cleanup=true
num_gselect=25
num_iters=5 # number of iters of transform estimation
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-dir> <vtln-dir> <exp-dir>"
   echo "e.g.: $0 data/train_novtln exp/vtln exp/train_warps"
   echo "where <vtln-dir> is produced by train_lvtln_model.sh"
   echo "Output is <exp-dir>/utt2warp"
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


if [ $stage -le -1 ]; then
  echo "$0: computing Gaussian selection info."

  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $vtlndir/final.ali_dubm "$sifeats" \
      "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

feats="$sifeats"

x=0
while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    echo "$0: pass $x of computing LVTLN transforms"
    
    if [ $x -eq 0 ]; then ubm=$vtlndir/final.ali_dubm; else ubm=$vtlndir/final.dubm; fi

    $cmd JOB=1:$nj $dir/log/lvtln.$x.JOB.log \
      gmm-global-gselect-to-post "$ubm" "$feats" \
        "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark:- \| \
      gmm-global-est-lvtln-trans \
        --logdet-scale=$logdet_scale --verbose=1 \
        $vtlndir/final.dubm $vtlndir/final.lvtln "$sifeats" ark,s,cs:- \
        ark:$dir/trans.$x.JOB ark,t:$dir/warp.$x.JOB || exit 1
  
    # consolidate the warps into one file.
    for j in $(seq $nj); do cat $dir/warp.$x.$j; done > $dir/warp.$x
    rm $dir/warp.$x.*
  fi
  feats="$sifeats transform-feats ark:$dir/trans.$x.JOB ark:- ark:- |"
  x=$[$x+1]
done

ln -sf warp.$[$x-1] $dir/utt2warp

$cleanup && rm $dir/gselect.*.gz $dir/trans.*
echo "$0: Distribution of classes for one job is below."
grep 'Distribution of classes' $dir/log/lvtln.$[$x-1].1.log


echo "$0: created warp factors in $dir/utt2warp"


# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

echo "$0: Done getting VTLN warps in $dir"
