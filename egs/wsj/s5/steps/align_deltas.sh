#!/bin/bash
# Copyright 2012  Daniel Povey
# Apache 2.0

# Computes training alignments using a model with delta features.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


nj=4
cmd=run.pl
oldgraphs=false
config=
for x in `seq 4`; do
  [ "$1" == --use-graphs ]  && oldgraphs=true && shift;
  [ "$1" == "--num-jobs" ] && nj=$2 && shift 2;
  [ "$1" == "--cmd" ] && cmd=$2 && shift 2;
  [ "$1" == "--config" ] && config=$2 && shift 2;
done


if [ $# != 4 ]; then
   echo "usage: steps/align_deltas.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_deltas.sh data/train data/lang exp/tri1 exp/tri1_ali"
   exit 1;
fi

[ -f path.sh ] && . ./path.sh

data=$1
lang=$2
srcdir=$3
dir=$4


# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
#End configuration.
[ ! -z $config ] && . $config

oov_sym=`cat $lang/oov.txt`

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl,final.occs} $dir || exit 1;

feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

echo "align_deltas.sh: aligning data in $data using model from $srcdir, putting alignments in $dir"

if $oldgraphs; then 
  [ $nj != "`cat $srcdir/num_jobs`" ] && echo "Mismatch in num-jobs" && exit 1;
  [ ! -f $srcdir/1.fsts.gz ] && echo "no such file $srcdir/1.fsts.gz" && exit 1;

  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl \
      "ark:gunzip -c $srcdir/$n.fsts.gz|" "$feats" "ark:|gzip -c >$dir/$n.ali.gz" || exit 1;
else
  tra="ark:utils/sym2int.pl --map-oov \"$oov_sym\" -f 2- $lang/words.txt $sdata/JOB/text|";
  # We could just use gmm-align in the next line, but it's less efficient as it compiles the
  # training graphs one by one.
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl ark:- \
      "$feats" "ark:|gzip -c >$dir/JOB.ali.gz" || exit 1;
fi

echo "Done aligning data."
