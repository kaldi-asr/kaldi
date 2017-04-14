#!/bin/bash

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey).
#           2015 David Snyder
# Apache 2.0.
#
# This script is based off of get_lda.sh in ../../steps/nnet2/, but has been
# modified for speaker recogntion purposes to use a sliding window CMN.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.

# Begin configuration section.
cmd=run.pl

feat_type=
stage=0
splice_width=4 # meaning +- 4 frames on each side for second LDA
left_context= # left context for second LDA
right_context= # right context for second LDA
rand_prune=4.0 # Relates to a speedup we do for LDA.
within_class_factor=0.0001 # This affects the scaling of the transform rows...
                           # sorry for no explanation, you'll have to see the code.
transform_dir=     # If supplied, overrides alidir
num_feats=10000 # maximum number of feature files to use.  Beyond a certain point it just
                # gets silly to use more data.
lda_dim=  # This defaults to no dimension reduction.
online_ivector_dir=
ivector_randomize_prob=0.0 # if >0.0, randomizes iVectors during training with
                           # this prob per iVector.
ivector_dir=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/nnet2/get_lda.sh [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/nnet2/get_lda.sh data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo " As well as extracting the examples, this script will also do the LDA computation,"
  echo " if --est-lda=true (default:true)"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --left-context <width;4>                         # Number of frames on left side to append for feature input, overrides splice-width"
  echo "  --right-context <width;4>                        # Number of frames on right side to append for feature input, overrides splice-width"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --online-vector-dir <dir|none>                   # Directory produced by"
  echo "                                                   # steps/online/nnet2/extract_ivectors_online.sh"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

[ -z "$left_context" ] && left_context=$splice_width
[ -z "$right_context" ] && right_context=$splice_width

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
oov=`cat $lang/oov.int`
num_leaves=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
cp $alidir/tree $dir

[ -z "$transform_dir" ] && transform_dir=$alidir

## Set up features.  Note: these are different from the normal features
## because we have one rspecifier that has the features for the entire
## training set, not separate ones for each batch.
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && ! [ -f $alidir/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"


# If we have more than $num_feats feature files (default: 10k),
# we use a random subset.  This won't affect the transform much, and will
# spare us an unnecessary pass over the data.  Probably 10k is
# way too much, but for small datasets this phase is quite fast.
N=$[$num_feats/$nj]

case $feat_type in
  raw) feats="ark,s,cs:utils/subset_scp.pl --quiet $N $sdata/JOB/feats.scp | apply-cmvn-sliding --center=true scp:- ark:- |"
   ;;
  lda)
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/{splice_opts,final.mat} $dir || exit 1;
     feats="ark,s,cs:utils/subset_scp.pl --quiet $N $sdata/JOB/feats.scp | apply-cmvn-sliding --center=true scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -f $transform_dir/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi
if [ -f $transform_dir/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/raw_trans.JOB ark:- ark:- |"
fi


feats_one="$(echo "$feats" | sed s:JOB:1:g)"
# note: feat_dim is the raw, un-spliced feature dim without the iVectors.
feat_dim=$(feat-to-dim "$feats_one" -) || exit 1;
# by default: no dim reduction.

spliced_feats="$feats splice-feats --left-context=$left_context --right-context=$right_context ark:- ark:- |"

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  # note: subsample-feats, with negative value of n, repeats each feature n times.
  spliced_feats="$spliced_feats paste-feats --length-tolerance=$ivector_period ark:- 'ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- | ivector-randomize --randomize-prob=$ivector_randomize_prob ark:- ark:- |' ark:- |"
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
else
  ivector_dim=0
fi
echo $ivector_dim >$dir/ivector_dim

if [ -z "$lda_dim" ]; then
  spliced_feats_one="$(echo "$spliced_feats" | sed s:JOB:1:g)"
  lda_dim=$(feat-to-dim "$spliced_feats_one" -) || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "$0: Accumulating LDA statistics."
  rm $dir/lda.*.acc 2>/dev/null # in case any left over from before.
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$rand_prune $alidir/final.mdl "$spliced_feats" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
fi

echo $feat_dim > $dir/feat_dim
echo $lda_dim > $dir/lda_dim
echo $ivector_dim > $dir/ivector_dim

if [ $stage -le 1 ]; then
  sum-lda-accs $dir/lda.acc $dir/lda.*.acc 2>$dir/log/lda_sum.log || exit 1;
  rm $dir/lda.*.acc
fi

if [ $stage -le 2 ]; then
  # There are various things that we sometimes (but not always) need
  # the within-class covariance and its Cholesky factor for, and we
  # write these to disk just in case.
  nnet-get-feature-transform --write-cholesky=$dir/cholesky.tpmat \
     --write-within-covar=$dir/within_covar.spmat \
     --within-class-factor=$within_class_factor --dim=$lda_dim \
      $dir/lda.mat $dir/lda.acc \
      2>$dir/log/lda_est.log || exit 1;
fi

echo "$0: Finished estimating LDA"
