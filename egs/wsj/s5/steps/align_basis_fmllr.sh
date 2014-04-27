#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2013  GoVivace Inc (Author: Nagendra Goel)
# Apache 2.0

# Computes training alignments; assumes features are (LDA+MLLT or delta+delta-delta)
# + fMLLR (probably with SAT models).
# It first computes an alignment with the final.alimdl (or the final.mdl if final.alimdl
# is not present), then does 2 iterations of fMLLR estimation.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match the source directory.


# Begin configuration section.  
stage=0
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.5 # factor by which to boost silence during alignment.
fmllr_update_type=full
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_fmllr.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_fmllr.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --fmllr-update-type (full|diag|offset|none)      # default full."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4
graphdir=$dir

oov=`cat $lang/oov.int` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;
alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/boost_phones.csl` $alimdl - |"
mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |"

## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1;
  graphdir=$srcdir
else
  graphdir=$dir
  if [ $stage -le 0 ]; then
    echo "$0: compiling training graphs"
    tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";   
    $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
      compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
  fi
fi


if [ $stage -le 1 ]; then
  echo "$0: aligning data in $data using $alimdl and speaker-independent features."
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$alimdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$sifeats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing fMLLR transforms"
  if [ "$alimdl" != "$mdl" ]; then
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
      gmm-est-basis-fmllr-gpost --fmllr-min-count=22  --num-iters=10 \
        --size-scale=0.2 --step-size-iters=3 \
        --write-weights=ark:$dir/pre_wgt.JOB \
        $mdl $srcdir/fmllr.basis "$sifeats"  ark,s,cs:- \
        ark:$dir/trans.JOB || exit 1;
#  else
#    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
#      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
#      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
#      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
#      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
#      ark,s,cs:- ark:$dir/trans.JOB || exit 1;
  fi
fi

feats="$sifeats transform-feats ark:$dir/pre_trans.JOB ark:- ark:- |"

if [ $stage -le 3 ]; then
  echo "$0: doing final alignment."
  $cmd JOB=1:$nj $dir/log/align_pass2.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

#rm $dir/pre_ali.*.gz

echo "$0: done aligning data."

utils/summarize_warnings.pl $dir/log

exit 0;
