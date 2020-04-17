#!/bin/bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#                2019  Yiming Wang
#  Apache 2.0.


# This script is modified from steps/nnet3/chain/build_tree.sh, but only contains
# trivial mono phone tree building without any states tying.


# Begin configuration section.
cmd=run.pl
frame_subsampling_factor=1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 --frame-subsampling-factor 3 \\"
  echo "    data/train data/lang_chain exp/mono_ali_train_sp exp/chain/tree"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --frame-subsampling-factor <factor>              # Factor (e.g. 3) controlling frame subsampling"
  echo "                                                   # at the neural net output, so the frame rate at"
  echo "                                                   # the output is less than at the input."
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir/delta_opts 2>/dev/null`

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
cp $alidir/delta_opts $dir 2>/dev/null # delta option.
cp $alidir/ali.1.gz $dir 2>/dev/null # to pass the file checking later during training

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs
if [ -f $alidir/per_utt ]; then
  sdata=$data/split${nj}utt
  utils/split_data.sh --per-utt $data $nj
else
  sdata=$data/split$nj
  utils/split_data.sh $data $nj
fi

# Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

## Set up speaker-independent features.
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# Add fMLLR transforms if available
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi

# Do subsampling of feats, if needed
if [ $frame_subsampling_factor -gt 1 ]; then
  feats="$feats subsample-feats --n=$frame_subsampling_factor ark:- ark:- |"
fi

echo "$0: Initializing monophone model (for alignment conversion, in case topology changed)"

[ ! -f $lang/phones/sets.int ] && exit 1;
shared_phones_opt="--shared-phones=$lang/phones/sets.int"
# get feature dimension
example_feats="`echo $feats | sed s/JOB/1/g`";
if ! feat_dim=$(feat-to-dim "$example_feats" - 2>/dev/null) || [ -z $feat_dim ]; then
  feat-to-dim "$example_feats" - # to see the error message.
  echo "error getting feature dimension"
  exit 1;
fi
$cmd JOB=1 $dir/log/init_mono.log \
  gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/mono.mdl $dir/mono.tree || exit 1;

cp $dir/mono.mdl $dir/final.mdl
cp $dir/mono.tree $dir/tree

echo $0: Done building tree
