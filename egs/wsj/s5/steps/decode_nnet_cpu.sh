#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script does decoding with a neural-net.  If the neural net was built on
# top of fMLLR transforms from a conventional system, you should provide the
# --transform-dir option.

# Begin configuration section.
stage=1
transform_dir=    # dir to find fMLLR transforms.
nj=4 # number of decoding jobs.  If --transform-dir set, must match that number!
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
cmd=run.pl
beam=15.0
max_active=7000
lat_beam=8.0 # Beam we use in lattice generation.
iter=final
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: steps/decode_nnet_cpu.sh [options] <graph-dir> <data-dir> <decode-dir>"
  echo " e.g.: steps/decode_nnet_cpu.sh --transform-dir exp/tri3b/decode_dev93_tgpr \\"
  echo "      exp/tri3b/graph_tgpr data/test_dev93 exp/tri4a_nnet/decode_dev93_tgpr"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/$iter.mdl

for f in $graphdir/HCLG.fst $data/feats.scp $model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

# Generate state-level lattice which we can rescore. 
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    nnet-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lat_beam \
    --acoustic-scale=$acwt --determinize-lattice=false --allow-partial=true \
    --word-symbol-table=$graphdir/words.txt "$model" \
    $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at 
# different acoustic scales to get the final output.


if [ $stage -le 2 ]; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  echo "score best paths"
  local/score.sh --cmd "$cmd" $data $graphdir $dir
  echo "score confidence and timing with sclite"
fi
echo "Decoding done."
exit 0;
