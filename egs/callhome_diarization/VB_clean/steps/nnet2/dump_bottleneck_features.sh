#!/bin/bash

#           2014  Pegah Ghahremani
# Apache 2.0


# Begin configuration section.
feat_type=
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
   echo "usage: steps/nnet2/dump_bottleneck_features.sh <input-data-dir> <output-data-dir> <bnf-nnet-dir> <archive-dir> <log-dir>"
   echo "e.g.:  steps/nnet2/dump_bottleneck_features.sh data/train data/train_bnf exp_bnf/bnf_net exp/tri5_ali mfcc exp_bnf/dump_bnf"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
bnf_data=$2
nnetdir=$3
archivedir=$4
dir=$5

# Assume that final.nnet is in nnetdir
bnf_nnet=$nnetdir/final.raw
if [ ! -f $bnf_nnet ] ; then
  echo "No such file $bnf_nnet";
  exit 1;
fi

## Set up input features of nnet
if [ -z "$feat_type" ]; then
  if [ -f $nnetdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
fi
echo "$0: feature type is $feat_type"

if [ "$feat_type" == "lda" ] && [ ! -f $nnetdir/final.mat ]; then
  echo "$0: no such file $nnetdir/final.mat"
  exit 1
fi

name=`basename $data`
sdata=$data/split$nj

mkdir -p $dir/log
mkdir -p $bnf_data
echo $nj > $nnetdir/num_jobs
splice_opts=`cat $nnetdir/splice_opts 2>/dev/null`
delta_opts=`cat $nnetdir/delta_opts 2>/dev/null`
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $nnetdir/final.mat ark:- ark:- |"
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then
  echo "Using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "No such file $transform_dir/trans.1" && exit 1;
  transform_nj=`cat $transform_dir/num_jobs` || exit 1;
  if [ "$nj" != "$transform_nj" ]; then
    for n in $(seq $transform_nj); do cat $transform_dir/trans.$n; done >$dir/trans.ark
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.ark ark:- ark:- |"
  else
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
  fi
fi


if [ $stage -le 1 ]; then
  echo "Making BNF scp and ark."
  $cmd JOB=1:$nj $dir/log/make_bnf_$name.JOB.log \
    nnet-compute $bnf_nnet "$feats" ark:- \| \
    copy-feats --compress=true ark:- ark,scp:$archivedir/raw_bnfeat_$name.JOB.ark,$archivedir/raw_bnfeat_$name.JOB.scp || exit 1;
fi

rm $dir/trans.ark 2>/dev/null

N0=$(cat $data/feats.scp | wc -l)
N1=$(cat $archivedir/raw_bnfeat_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "Error happens when generating BNF for $name (Original:$N0  BNF:$N1)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in $(seq $nj); do  cat $archivedir/raw_bnfeat_$name.$n.scp; done > $bnf_data/feats.scp

for f in segments spk2utt text utt2spk wav.scp char.stm glm kws reco2file_and_channel stm; do
  [ -e $data/$f ] && cp -r $data/$f $bnf_data/$f
done

echo "$0: computing CMVN stats."
steps/compute_cmvn_stats.sh $bnf_data $dir $archivedir

echo "$0: done making BNF feats.scp."

exit 0;
