#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This is to compute validation likelihood on a different set,
# not sampled from the regular training set.


# Begin configuration section.
cmd=run.pl
num_valid_frames=10000 # a subset of the frames in "valid_utts".
iter=final
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: steps/eval_nnet_like.sh <data> <ali-dir> <exp-dir>"
  echo " e.g.: steps/eval_nnet_like.sh data/train_dev exp/tri3a_ali_train_dev exp/tri4a_nnet"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-valid-frames <n>                           # number of validation frames to sample (default: 10k)"
  echo "  --iter <iter>                                    # iteration to test (default: final)"
  exit 1;
fi

data=$1
alidir=$2
dir=$3

# Check some files.
for f in $data/feats.scp $dir/${iter}.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log

## Set up features.   Note:

if [ -f $dir/final.mat ]; then 
  feat_type=lda; 
  cmp $dir/final.mat $alidir/final.mat || exit 1;
else 
  feat_type=delta; 
fi
echo "$0: feature type is $feat_type"

splice_opts=`cat $alidir/splice_opts 2>/dev/null`

case $feat_type in
  delta) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
   ;;
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
fi

name=`basename $data`

echo "Evaluating likelihood"
$cmd $dir/log/eval_like_${name}_it${iter}.log \
  nnet-randomize-frames --num-samples=$num_valid_frames \
     --srand=0 "$feats" "ark,cs:gunzip -c $alidir/ali.*.gz | ali-to-pdf $dir/${iter}.mdl ark:- ark:- |" \
    ark:- \| \
   nnet-compute-prob $dir/${iter}.mdl ark:-  || exit 1;


echo Done
