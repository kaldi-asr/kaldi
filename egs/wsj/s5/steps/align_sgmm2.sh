#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments and (if needed) speaker-vectors, given an 
# SGMM system.  If the system is built on top of SAT, you should supply
# transforms with the --transform-dir option.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory.

# Begin configuration section.  
stage=0
nj=4
cmd=run.pl
use_graphs=false # use graphs from srcdir
use_gselect=false # use gselect info from srcdir [regardless, we use
   # Gaussian-selection info, we might have to compute it though.]
gselect=15  # Number of Gaussian-selection indices for SGMMs.
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
transform_dir=  # directory to find fMLLR transforms in.
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_sgmm.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_sgmm.sh --transform-dir exp/tri3b data/train data/lang \\"
   echo "           exp/sgmm4a exp/sgmm5a_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --transform-dir <transform-dir>                  # directory to find fMLLR transforms"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
sdata=$data/split$nj

mkdir -p $dir/log
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
[ -f $srcdir/final.alimdl ] && cp $srcdir/final.alimdl $dir
cp $srcdir/final.occs $dir;

## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option during alignment."
fi
##

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;

## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1;
  graphdir=$srcdir
  ln.pl $srcdir/fsts.*.gz $dir
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

## Work out where we're getting the Gaussian-selection info from
if $use_gselect; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-gselect true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/gselect.1.gz ] && echo "No gselect info in $srcdir" && exit 1;
  graphdir=$srcdir
  gselect_opt="--gselect=ark,s,cs:gunzip -c $srcdir/gselect.JOB.gz|"
  ln.pl $srcdir/gselect.*.gz $dir
else
  graphdir=$dir
  if [ $stage -le 1 ]; then
    echo "$0: computing Gaussian-selection info"
    # Note: doesn't matter whether we use $alimdl or $mdl, they will
    # have the same gselect info.
    $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
      sgmm2-gselect --full-gmm-nbest=$gselect $alimdl \
      "$feats" "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
  fi
  gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"
fi


if [ $alimdl == $mdl ]; then 
  # Speaker-independent decoding-- just one pass.  Not normal.
  T=`sgmm2-info $mdl | grep 'speaker vector space' | awk '{print $NF}'` || exit 1;
  [ "$T" -ne 0 ] && echo "No alignment model, yet speaker vector space nonempty" && exit 1;

  if [ $stage -le 2 ]; then
    echo "$0: aligning data in $data using model $mdl (no speaker-vectors)"
    $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
      sgmm2-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $alimdl \
      "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi
  echo "$0: done aligning data."
  exit 0;
fi

# Continue with system with speaker vectors.
if [ $stage -le 2 ]; then
  echo "$0: aligning data in $data using model $alimdl"
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    sgmm2-align-compiled $scale_opts "$gselect_opt" --beam=$beam --retry-beam=$retry_beam $alimdl \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: computing speaker vectors (1st pass)"
  $cmd JOB=1:$nj $dir/log/spk_vecs1.JOB.log \
    ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
    sgmm2-post-to-gpost "$gselect_opt" $alimdl "$feats" ark:- ark:- \| \
    sgmm2-est-spkvecs-gpost --spk2utt=ark:$sdata/JOB/spk2utt \
     $mdl "$feats" ark,s,cs:- ark:$dir/pre_vecs.JOB || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: computing speaker vectors (2nd pass)"
  $cmd JOB=1:$nj $dir/log/spk_vecs2.JOB.log \
    ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
    sgmm2-est-spkvecs --spk2utt=ark:$sdata/JOB/spk2utt "$gselect_opt" \
     --spk-vecs=ark:$dir/pre_vecs.JOB $mdl "$feats" ark,s,cs:- ark:$dir/vecs.JOB || exit 1;
  rm $dir/pre_vecs.*
fi

if [ $stage -le 5 ]; then
  echo "$0: doing final alignment."
  $cmd JOB=1:$nj $dir/log/align_pass2.JOB.log \
    sgmm2-align-compiled $scale_opts "$gselect_opt" --beam=$beam --retry-beam=$retry_beam \
     --utt2spk=ark:$sdata/JOB/utt2spk --spk-vecs=ark:$dir/vecs.JOB \
     $mdl "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

rm $dir/pre_ali.*.gz

echo "$0: done aligning data."

utils/summarize_warnings.pl $dir/log

exit 0;
