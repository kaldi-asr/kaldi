#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Computes training alignments using log-likelihoods 

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


# Begin configuration section.
nj=4
cmd=run.pl
graphs=
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
careful=false
boost_silence=1.0 # Factor by which to boost silence during alignment.
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: steps/align_mapped.sh <data-dir> <prob-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_mapped.sh data/test exp/prob_test data/lang exp/nnet3 exp/ali_test"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --graphs <dir>                                   # set graphs dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
prob=$2
lang=$3
srcdir=$4
dir=$5


for f in $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split${nj}

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;

feats="ark:$prob/output.JOB.ark"

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

mdl=$dir/final.mdl

if [ ! -z "$graphs" ]; then
  if [ ! -d $graphs ]; then
    echo "Could not find graphs $graphs" && exit 1
  fi

  [ $nj != "`cat $graphs/num_jobs`" ] && echo "$0: mismatch in num-jobs" && exit 1;
  [ ! -f $graphs/fsts.1.gz ] && echo "$0: no such file $graphs/fsts.1.gz" && exit 1;

  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $graphs/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
else
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  # We could just use gmm-align in the next line, but it's less efficient as it compiles the
  # training graphs one by one.
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

echo "$0: done aligning data."
