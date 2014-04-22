#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# mix up (or down); do 3 iters of model training; realign; then do two more
# iterations of model training.

# Begin configuration section.
cmd=run.pl
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=5
realign_iters=3 # Space-separated list of iterations to realign on.
stage=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/mixup.sh <num-gauss> <data-dir> <lang-dir> <old-exp-dir> <exp-dir>"
   echo " e.g.: steps/mixup.sh 20000 data/train_si84 data/lang exp/tri3b exp/tri3b_20k"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi

numgauss=$1
data=$2
lang=$3
srcdir=$4
dir=$5

for f in $data/feats.scp $srcdir/final.mdl $srcdir/final.mat; do
  [ ! -f $f ] && echo "mixup_lda_etc.sh: no such file $f" && exit 1;
done

nj=`cat $srcdir/num_jobs` || exit 1;
sdata=$data/split$nj;

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

mkdir -p $dir/log
cp $srcdir/splice_opts $dir 2>/dev/null
cp $srcdir/cmvn_opts $dir 2>/dev/null
cp $srcdir/final.mat $dir
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/tree $dir


## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ -f $srcdir/trans.1 ]; then
  echo Using transforms from $srcdir;
  rm $dir/trans.* 2>/dev/null
  ln.pl $srcdir/trans.* $dir  # Link those transforms to current directory.
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
else
  feats="$sifeats"
fi
## Done setting up features.

rm $dir/fsts.*.gz 2>/dev/null
ln.pl $srcdir/fsts.*.gz $dir  # Link training-graph FSTs to current directory.

## Mix up old model
if [ $stage -le 0 ]; then
  echo Mixing up old model to $numgauss Gaussians
# Note: this script also works for mixing down.
  $cmd $dir/log/mixup.log \
    gmm-mixup --mix-up=$numgauss --mix-down=$numgauss \
    $srcdir/final.mdl $srcdir/final.occs $dir/1.mdl || exit 1;
fi
## Done.

cur_alidir=$srcdir # dir to find alignments.
[ -z "$realign_iters" ] && ln.pl $srcdir/ali.*.gz $dir; # link alignments, if
 # we won't be generating them.

x=1
while [ $x -le $num_iters ]; do
  echo "$0: iteration $x"
  if echo $realign_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "$0: realigning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
        "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi
    cur_alidir=$dir
  fi
  if [ $stage -le $x ]; then
    echo "$0: accumulating statistics"
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $cur_alidir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    echo "$0: re-estimating model"
    [ "`ls $dir/$x.*.acc | wc -w`" -ne $nj ] && echo "$0: wrong #accs" && exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc
    rm $dir/$x.occs  2>/dev/null
  fi
  x=$[$x+1]
done

rm $dir/final.mdl $dir/final.occs 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs

if [ -f $dir/trans.1 ]; then 
  echo "$0: accumulating stats for alignment model."
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1;
  [ "`ls $dir/$x.*.acc | wc -w`" -ne $nj ] && echo "$0: wrong #accs" && exit 1;  
  echo "$0: Re-estimating alignment model."
  $cmd $dir/log/est_alimdl.log \
    gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl  || exit 1;
  rm $dir/$x.*.acc
  rm $dir/final.alimdl 2>/dev/null
  ln -s $x.alimdl $dir/final.alimdl 
fi

utils/summarize_warnings.pl $dir/log

echo Done
