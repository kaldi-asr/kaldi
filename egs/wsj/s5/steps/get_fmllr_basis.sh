#!/usr/bin/env bash

# Copyright 2012   Carnegie Mellon University (Author: Yajie Miao)
#                  Johns Hopkins University (Author: Daniel Povey)

# Decoding script that computes basis for basis-fMLLR (see decode_fmllr_basis.sh).
# This can be on top of delta+delta-delta, or LDA+MLLT features.

stage=0
# Parameters in alignment of training data
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
per_utt=true # If true, then treat each utterance as a separate speaker for purposes of
  # basis training... this is recommended if the number of actual speakers in your
  # training set is less than (feature-dim) * (feature-dim+1).
silence_weight=0.01
cmd=run.pl
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/get_fmllr_basis.sh [options] <data-dir> <lang-dir> <exp-dir>"
   echo " e.g.: steps/decode_basis_fmllr.sh data/train_si84 data/lang exp/tri3b/"
   echo "Note: we currently assume that this is the same data you trained the model with."
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

data=$1
lang=$2
dir=$3

nj=`cat $dir/num_jobs` || exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

for f in $data/feats.scp $dir/final.mdl $dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

utils/lang/check_phones_compatible.sh $lang/phones.txt $dir/phones.txt || exit 1;
# Set up the unadapted features "$sifeats".
if [ -f $dir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

# Set up the adapted features "$feats" for training set.
if [ -f $srcdir/trans.1 ]; then 
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$srcdir/trans.JOB ark:- ark:- |";
else
  feats="$sifeats";
fi


if $per_utt; then
  spk2utt_opt=  # treat each utterance as separate speaker when computing basis.
  echo "Doing per-utterance adaptation for purposes of computing the basis."
else
  echo "Doing per-speaker adaptation for purposes of computing the basis."
  [ `cat $sdata/spk2utt | wc -l` -lt $[41*40] ] && \
    echo "Warning: number of speakers is small, might be better to use --per-utt=true."
  spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi

# Note: we get Gaussian level alignments with the "final.mdl" and the
# speaker adapted features. 
$cmd JOB=1:$nj $dir/log/basis_acc.JOB.log \
  ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
  weight-silence-post $silence_weight $silphonelist $dir/final.mdl ark:- ark:- \| \
  gmm-post-to-gpost $dir/final.mdl "$feats" ark:- ark:- \| \
  gmm-basis-fmllr-accs-gpost $spk2utt_opt \
    $dir/final.mdl "$sifeats" ark,s,cs:- $dir/basis.acc.JOB || exit 1; 

# Compute the basis matrices.
$cmd $dir/log/basis_training.log \
  gmm-basis-fmllr-training $dir/final.mdl $dir/fmllr.basis $dir/basis.acc.* || exit 1;
rm $dir/basis.acc.* 2>/dev/null

exit 0;

