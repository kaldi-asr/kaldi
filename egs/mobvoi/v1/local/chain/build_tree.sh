#!/bin/bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#  Apache 2.0.


# This script builds a tree for use in the 'chain' systems (although the script
# itself is pretty generic and doesn't use any 'chain' binaries).  This is just
# like the first stages of a standard system, like 'train_sat.sh', except it
# does 'convert-ali' to convert alignments to a monophone topology just created
# from the 'lang' directory (in case the topology is different from where you
# got the system's alignments from), and it stops after the tree-building and
# model-initialization stage, without re-estimating the Gaussians or training
# the transitions.


# Begin configuration section.
stage=-5
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported.
cmd=run.pl
frame_subsampling_factor=1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 --frame-subsampling-factor 3 \\"
  echo "    data/train_si84 data/lang_chain exp/tri3b_ali_si284_sp exp/chain/tree_a_sp"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --repeat-frames <true|false>                     # Only affects alignment conversion at"
  echo "                                                   # the end. If true, generate an "
  echo "                                                   # alignment using the frame-subsampled "
  echo "                                                   # topology that is repeated "
  echo "                                                   # --frame-subsampling-factor times "
  echo "                                                   # and interleaved, to be the same "
  echo "                                                   # length as the original alignment "
  echo "                                                   # (useful for cross-entropy training "
  echo "                                                   # of reduced frame rate systems)."
  echo "  --context-opts <option-string>                   # Options controlling phonetic context;"
  echo "                                                   # we suggest '--context-width=2 --central-position=1',"
  echo "                                                   # which is left bigram."
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

if [ -z $alignment_subsampling_factor ]; then
  alignment_subsampling_factor=$frame_subsampling_factor
fi

if [ $stage -le -5 ]; then
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
fi

cp $dir/mono.mdl $dir/final.mdl
cp $dir/mono.tree $dir/tree

echo $0: Done building tree
