#!/bin/bash
# Copyright 2013  Carnegie Mellon University (Author: Yajie Miao)
# Apache 2.0

# Create pfiles for deep neural network training
# Assumes the training alignment is ready and features are (LDA+MLLT or delta+delta-delta)
# + fMLLR
# 

# We splice 9 frames (+4, -4) of LDA+MLLT+fMLLR features, which have the dimension of 360 (if 
# we use the default LDA dim of 40).
# Then, an LDA is estimated and applied, projecting the spliced features down to the
# dimension of [nnet_dim] (250 by default)

# You can specify the neighbour size by --nnet-splice-opts and the final dimension by
# --nnet-dim


# Begin configuration section.  
stage=1
nj=4
cmd=run.pl

# Begin configuration.
nnet_splice_opts="--left-context=4 --right-context=4" # frame-splicing options for nnet input
nnet_dim=250 # the dimension after LDA, i.e., the dimension of the input for DNN
pfile_unit_size=40 # the number of utterances of each small unit into which 
#the whole (and huge) pfile is chopped 
cv_ratio=0.05 # the ratio of CV data

boost_silence=1.0 # factor by which to boost silence during alignment.
randprune=4.0
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/build_nnet_pfile.sh <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo "e.g.:  steps/build_nnet_pfile.sh data/train data/lang exp/tri4_ali exp/tri4_pfile"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --nnet-splice-opts                               # how frames are spliced for DNN input"
   echo "  --nnet-dim                                        # the final [# of dim] after LDA"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
echo $nnet_splice_opts > $dir/nnet_splice_opts
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Setup features (prior to LDA)
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  sifeats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
# Now the spliced features for LDA
splicedfeats="$sifeats splice-feats $nnet_splice_opts ark:- ark:- |"
##

if [ $stage -le 1 ]; then
  echo "Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
  est-lda --dim=$nnet_dim $dir/final.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
fi

# Now the feats input to nnet
feats="$splicedfeats transform-feats $dir/final.mat ark:- ark:- |"

if [ $stage -le 2 ]; then
  echo "Generate nnet training pfiles"
  $cmd JOB=1:$nj $dir/log/build_pfile.JOB.log \
    build-pfile-from-ali $alidir/final.mdl "ark:gunzip -c $alidir/ali.JOB.gz|" \
      "$feats" "|pfile_create -i - -o $dir/pfile.JOB -f $nnet_dim -l 1" || exit 1;
  # concatenate the pfiles into one
  all_pfiles=""
  for n in `seq 1 $nj`; do
    all_pfiles="$all_pfiles $dir/pfile.$n"
  done
  pfile_concat $all_pfiles -o $dir/concat.pfile 2>$dir/log/pfile_cat.log || exit 1;
  rm $dir/pfile.*
fi

if [ $stage -le 3 ]; then
  echo "Split data into training and cross-validation"
  mkdir -p $dir/concat
  # Chop the whole pfile into small units
  perl steps_BNF/pfile_burst.pl -i $dir/concat.pfile -o $dir/concat -s $pfile_unit_size \
     2>$dir/log/pfile_burst.log || exit 1;

  # Split the units accoring to cv_ratio
  perl steps_BNF/pfile_rconcat.pl -o $dir/valid.pfile,${cv_ratio} -o $dir/train.pfile \ 
     $dir/concat/*.pfile 2>$dir/log/pfile_rconcat.log || exit 1;
  rm -r $dir/concat
  echo "## Info of the training pfile: ##"
  pfile_info $dir/train.pfile
  echo "## Info of the cross-validation pfile: ##"
  pfile_info $dir/valid.pfile

  # Compress the two (if everything is correct) pfiles 
  gzip $dir/train.pfile $dir/valid.pfile
fi

echo "$0: done creating pfiles."

exit 0;
