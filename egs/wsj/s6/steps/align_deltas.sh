#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments using a model with delta features, with no CMVN.

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.

# Begin configuration section.  
nj=4
cmd=run.pl
# Begin configuration.
use_graphs=false
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence during alignment. (1.0 does nothing)
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_deltas.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_deltas.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj

split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;

[ -f $srcdir/final.mat ] && echo "$0: you have an LDA matrix." && exit 1;

feats="ark,s,cs:add-deltas scp:$sdata/JOB/feats.scp ark:- |"

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

if [ $boost_silence != 1.0 ]; then
  mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/final.mdl - |"
else
  mdl=$dir/final.mdl
fi

if $use_graphs; then 
  [ $nj != "`cat $srcdir/num_jobs`" ] && echo "$0: mismatch in num-jobs" && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "$0: no such file $srcdir/fsts.1.gz" && exit 1;

  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
      "ark:gunzip -c $srcdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
else
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  # We could just use gmm-align in the next line, but it's less efficient as it compiles the
  # training graphs one by one.
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

utils/summarize_warnings.pl $dir/log

echo $0: Done
