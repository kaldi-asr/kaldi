#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
nj=4
cmd=run.pl
maxactive=7000
beam=15.0
lattice_beam=8.0
expand_beam=30.0
acwt=1.0
skip_scoring=false

stage=0
online_ivector_dir=
post_decode_acwt=10.0
extra_left_context=0
extra_right_context=0
extra_left_context_initial=0
extra_right_context_final=0
chunk_width=140,100,160
use_gpu=no
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/decode_biglm_fast.sh [options] <graph-dir> <old-LM-fst> <new-LM-fst> <data-dir> <decode-dir>"
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
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


graphdir=$1
oldlm_fst=$2
newlm_fst=$3
data=$4
dir=$5

srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $srcdir/final.mdl $graphdir/HCLG.fst $oldlm_fst $newlm_fst; do
  [ ! -f $f ] && echo "decode_si.sh: no such file $f" && exit 1;
done


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode_si.sh: feature type is $feat_type"

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

posteriors="ark,scp:$sdata/JOB/posterior.ark,$sdata/JOB/posterior.scp"
posteriors_scp="scp:$sdata/JOB/posterior.scp"

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
fi

frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
# generate log-likelihood
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/nnet_compute.JOB.log \
    nnet3-compute $ivector_opts $frame_subsampling_opt \
    --acoustic-scale=$acwt \
    --extra-left-context=$extra_left_context \
    --extra-right-context=$extra_right_context \
    --extra-left-context-initial=$extra_left_context_initial \
    --extra-right-context-final=$extra_right_context_final \
    --frames-per-chunk=$frames_per_chunk \
    --use-gpu=$use_gpu --use-priors=true \
    $srcdir/final.mdl "$feats" "$posteriors"
fi

[ -f `dirname $oldlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
  echo "Warning: old LM words.txt does not match with that in $graphdir .. probably will not work.";
[ -f `dirname $newlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
  echo "Warning: new LM words.txt does not match with that in $graphdir .. probably will not work.";

# fstproject replaces the disambiguation symbol #0, which only appears on the
# input side, with the <eps> that appears in the corresponding arcs on the output side.
oldlm_cmd="fstproject --project_output=true $oldlm_fst | fstarcsort --sort_type=ilabel |"
newlm_cmd="fstproject --project_output=true $newlm_fst | fstarcsort --sort_type=ilabel |"

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 latgen-biglm-faster-combine-mapped --max-active=$maxactive --beam=$beam \
   --lattice-beam=$lattice_beam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$oldlm_cmd" "$newlm_cmd" "$posteriors_scp" \
  "$lat_wspecifier" || exit 1;

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
