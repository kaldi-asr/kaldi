#!/bin/bash

# Copyright 2012  Daniel Povey
# Apache 2.0

# Begin configuration section.  
iter=
nj=4
cmd=run.pl
maxactive=7000
beam=13.0
latbeam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode_si.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode_si.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z $iter ]; then model=$srcdir/final.mdl; 
else model=$srcdir/$iter.mdl; fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode_si.sh: no such file $f" && exit 1;
done

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode_si.sh: feature type is $feat_type";

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-faster --max-active=$maxactive --beam=$beam --lattice-beam=$latbeam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $graphdir $dir

exit 0;
