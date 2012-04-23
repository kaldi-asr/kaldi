#!/bin/bash

# Copyright 2012  Daniel Povey
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh; # source the path.

nj=4
cmd=run.pl
config=

for x in `seq 3`; do
  [ $1 == "--num-jobs" ] && nj=$2 && shift 2;
  [ $1 == "--cmd" ] && cmd=$2 && shift 2;
  [ $1 == "--config" ] && config=$2 && shift 2;
done

if [ $# != 3 ]; then
   echo "Usage: steps/decode_deltas.sh <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode_deltas.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
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

# Begin configuration.
maxactive=7000
beam=13.0
latbeam=6.0
acwt=0.083333
# End configuration.
[ ! -z $config ] && . $config # Override any of the above, if --config specified.

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $srcdir/final.mdl $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode_deltas.sh: no such file $f" && exit 1;
done


feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-faster --max-active=$maxactive --beam=$beam --lattice-beam=$latbeam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $graphdir $dir

exit 0;
