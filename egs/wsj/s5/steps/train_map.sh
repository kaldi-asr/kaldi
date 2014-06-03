#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# Train a model on top of existing features (no feature-space learning of any
# kind is done).  This script does not re-train the tree, it just does one iteration
# of MAP adaptation to the model in the input alignment-directory.  It's useful for
# adapting a system to a specific gender, or new acoustic conditions.


# Begin configuration..
cmd=run.pl
stage=0
tau=20 # smoothing constant used in MAP estimation, corresponds to the number of 
       # "fake counts" that we add for the old model.  Larger tau corresponds to less
       # aggressive re-estimation, and more smoothing.  You might want to try 10 or 15 also
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: steps/train_map.sh <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_map.sh data/train_si84_female data/lang exp/tri3c_ali_si84_female exp/tri4b_female"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Set various variables.
nj=`cat $alidir/num_jobs` || exit 1;
sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`


mkdir -p $dir/log

cp $alidir/tree $dir
# link ali.*.gz from $alidir to dest directory.
utils/ln.pl $alidir/ali.*.gz $dir


echo $nj >$dir/num_jobs
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  ln.pl $alidir/trans.* $dir # Link them to dest dir.
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
else
  feats="$sifeats"
fi
##

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/acc.JOB.log \
    gmm-acc-stats-ali  $alidir/final.mdl "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz|"  $dir/0.JOB.acc || exit 1;
  
  [ "`ls $dir/0.*.acc | wc -w`" -ne "$nj" ] && echo "$0: wrong #accs" && exit 1;

  $cmd $dir/log/sum_accs.log \
    gmm-sum-accs $dir/0.acc $dir/0.*.acc || exit 1;

  rm $dir/0.*.acc
fi

if [ $stage -le 1 ]; then
  # Update only the model means.  This is traditional in MAP estimation.
  $cmd $dir/log/update.log \
     gmm-ismooth-stats --smooth-from-model --tau=$tau $alidir/final.mdl $dir/0.acc - \| \
     gmm-est --update-flags=m --write-occs=$dir/final.occs --remove-low-count-gaussians=false \
           $alidir/final.mdl - $dir/final.mdl || exit 1;
fi

echo Done
