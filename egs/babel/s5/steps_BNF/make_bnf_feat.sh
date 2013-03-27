#!/bin/bash
# Copyright 2013  Carnegie Mellon University (Author: Yajie Miao)
# Apache 2.0

# Make BNF front-end with the trained neural network
# Assumes the nnet inputs features are (LDA+MLLT or
# delta+delta-delta) + fMLLR
# 

# Begin configuration section.  
stage=1
nj=4
cmd=run.pl

# Begin configuration.
transform_dir=

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: steps/make_bnf_feat.sh <data-dir> <bnfdata-dir> <bnf-nnet-dir> <ali-dir> <exp-dir>"
   echo "e.g.:  steps/make_bnf_feat.sh data/train data/train_bnf exp/bnf_net exp/tri4_ali exp/make_bnf"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
bnf_data=$2
nnetdir=$3
alidir=$4
dir=$5

# because we [cat trans.*], no need to keep nj consistent with [# of trans]
nj=`cat $transform_dir/num_jobs` || exit 1;

# Assume that final.mat and final.nnet are at nnetdir
nnet_lda=$nnetdir/final.mat
bnf_nnet=$nnetdir/final.nnet
for file in $nnet_lda $bnf_nnet; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

name=`basename $data`
sdata=$data/split$nj

mkdir -p $dir/log
mkdir -p $bnf_data
echo $nj > $dir/num_jobs
nnet_splice_opts=`cat $nnetdir/nnet_splice_opts 2>/dev/null`
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Set up input features of nnet
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "Using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "No such file $transform_dir/trans.1" && exit 1;
#  cat $transform_dir/trans.* > $dir/trans || exit 1;
  sifeats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
fi
# Now apply the LDA 
feats="$sifeats splice-feats $nnet_splice_opts ark:- ark:- | transform-feats $nnet_lda ark:- ark:- |"
##

if [ $stage -le 1 ]; then
  echo "Making BNF scp and ark."
  $cmd JOB=1:$nj $dir/log/make_bnf.JOB.log \
    nnet-forward --apply-log=false $bnf_nnet "$feats" \
    ark,scp:$dir/raw_bnfeat_$name.JOB.ark,$dir/raw_bnfeat_$name.JOB.scp
fi

N0=$(cat $data/feats.scp | wc -l) 
N1=$(cat $dir/raw_bnfeat_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "Error happens when generating BNF for $name (Original:$N0  BNF:$N1)"
  exit 1;
fi

echo -n >$bnf_data/feats.scp
# Concatenate feats.scp into bnf_data
for n in `seq 1 $nj`; do
  cat $dir/raw_bnfeat_$name.$n.scp >> $bnf_data/feats.scp
done

for f in segments spk2utt text utt2spk wav.scp char.stm glm kws reco2file_and_channel stm; do
  [ -e $data/$f ] && cp -r $data/$f $bnf_data/$f
done

echo "$0: done making BNF feats.scp."

exit 0;
