#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Korbinian Riedhammer
# Apache 2.0

# Computes training alignments using a model with delta or
# LDA+MLLT features.

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


# Begin configuration section.  
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence during alignment.
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: steps/tandem/align_si.sh <data1-dir> <data2-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/tandem/align_si.sh {mfcc,bottleneck}/data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data1=$1
data2=$2
lang=$3
srcdir=$4
dir=$5

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs

# Set up the features

sdata1=$data1/split$nj
sdata2=$data2/split$nj
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;

# Get some info on the feature types
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $srcdir/normft2 2>/dev/null` || exit 1;

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

# for lda-type features, we need to copy both the lda (for baseft) and mllt 
# transformation (for the pasted features)
case $feat_type in
  delta) 
  	echo "$0: feature type is $feat_type"
	  ;;
  lda) 
  	echo "$0: feature type is $feat_type"
    cp $srcdir/{lda,final}.mat $dir/ || exit 1;
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features, 
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $srcdir/{tandem,splice_opts,normft2} $dir 2>/dev/null

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir"

mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/final.mdl - |"

if $use_graphs; then 
  [ $nj != "`cat $srcdir/num_jobs`" ] && echo "$0: mismatch in num-jobs" && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "$0: no such file $srcdir/fsts.1.gz" && exit 1;

  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
      "ark:gunzip -c $srcdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
else
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata1/JOB/text|";
  # We could just use gmm-align in the next line, but it's less efficient as it compiles the
  # training graphs one by one.
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

echo "$0: done aligning data."
