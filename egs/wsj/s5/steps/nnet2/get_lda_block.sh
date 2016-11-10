#!/bin/bash

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.

# Begin configuration section.
cmd=run.pl

stage=0
splice_width=4 # meaning +- 4 frames on each side for second LDA
rand_prune=4.0 # Relates to a speedup we do for LDA.
within_class_factor=0.0001 # This affects the scaling of the transform rows...
                           # sorry for no explanation, you'll have to see the code.
block_size=10
block_shift=5

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/nnet2/get_lda_block.sh [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/nnet2/get_lda.sh data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo " As well as extracting the examples, this script will also do the LDA computation,"
  echo " if --est-lda=true (default:true)"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree; do
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
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

## Set up features.  Note: these are different from the normal features
## because we have one rspecifier that has the features for the entire
## training set, not separate ones for each batch.


feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"
train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"

feat_dim=`feat-to-dim "$train_subset_feats" -` || exit 1;

if [ $stage -le 0 ]; then
  echo "$0: Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    set -o pipefail '&&' \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$rand_prune $alidir/final.mdl "$feats splice-feats --left-context=$splice_width --right-context=$splice_width ark:- ark:- |" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
fi

echo $feat_dim > $dir/feat_dim

echo -n > $dir/indexes
# Get list of indexes, e.g. a file like:
# 0 1 2 3 4 5 6 7 8 9
# 5 6 7 8 9 10 11 12 13 14
# 10 ...

cur_index=0
num_blocks=0
context_length=$[1+2*($splice_width)]

while true; do
  for n in `seq $cur_index $[cur_index+$block_size-1]`; do
    echo -n `seq $n $feat_dim $[$n+($feat_dim*($context_length-1))]` '' >> $dir/indexes
  done
  echo >> $dir/indexes
  num_blocks=$[$num_blocks+1]
  next_index=$[$cur_index+$block_shift]
  if [ $[$next_index+$block_size] -gt $feat_dim ]; then
    next_index=$[$feat_dim-$block_size];
  fi
  if [ $next_index -le $cur_index ]; then break; fi
  cur_index=$next_index
done
echo $num_blocks >$dir/num_blocks

lda_dim=`cat $dir/indexes | wc -w`
echo $lda_dim > $dir/lda_dim

if [ $stage -le 1 ]; then
  nnet-get-feature-transform-multi --within-class-factor=$within_class_factor $dir/indexes $dir/lda.*.acc $dir/lda.mat \
      2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
fi

echo "$0: Finished estimating LDA"
