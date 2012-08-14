#!/bin/bash

# Copyright 2012  Karel Vesely, Daniel Povey
# Apache 2.0

# Begin configuration section.  
iter=
nnet= # You can specify the nnet to use (e.g. if you want to use the .alinnet)
model= # You can specify the transition model to use (e.g. if you want to use the .alimdl)

nj=4
cmd=run.pl
max_active=7000
beam=13.0
latbeam=6.0
acwt=0.12 # note: only really affects pruning (scoring is on lattices).
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: $0 exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --nnet <nnet>                                    # which nnet to use (e.g. to"
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.nnet)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "                                                   # speaker-adapted decoding"
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

if [ -z "$nnet" ]; then # if --nnet <nnet> was not specified on the command line...
  if [ -z $iter ]; then nnet=$srcdir/final.nnet; 
  else nnet=$(find $srcdir/nnet/ -name nnet_*_iter{,0}${iter}_lrate*); fi
fi
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  model=$srcdir/final.mdl;
fi

#hard-select feature-extraction files
hamm_dct=$srcdir/hamm_dct.mat
cmvn_g=$srcdir/cmvn_glob.mat

#remove the softmax from the nnet
nnet_i=$nnet; nnet=$dir/$(basename $nnet)_nosoftmax;
nnet-trim-n-last-transforms --n=1 --binary=false $nnet_i $nnet 2>$dir/trim_softmax.log || exit 1;

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $hamm_dct $cmvn_g $nnet_i $nnet $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


feat_type=nnet
echo "$0: feature type is $feat_type";

norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
splice_opts=$(cat $srcdir/splice_opts 2>/dev/null)

case $feat_type in
  nnet) feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $hamm_dct ark:- ark:- | apply-cmvn --norm-vars=true $cmvn_g ark:- ark:- | nnet-forward --no-softmax=true --class-frame-counts=$srcdir/ali_train.counts $nnet ark:- ark:- |" ;;
  *) echo "Invalid feature type $feat_type" && exit 1 ;;
esac

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $graphdir $dir

exit 0;
