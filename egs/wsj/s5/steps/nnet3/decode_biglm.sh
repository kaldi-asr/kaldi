#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.
stage=1
nj=4 # number of decoding jobs.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
cmd=run.pl
beam=15.0
frames_per_chunk=50
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0 # Beam we use in lattice generation.
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
use_gpu=false # If true, will use a GPU, with nnet3-latgen-faster-batch.
              # In that case it is recommended to set num-threads to a large
              # number, e.g. 20 if you have that many free CPU slots on a GPU
              # node, and to use a small number of jobs.
scoring_opts=
skip_diagnostics=false
skip_scoring=false
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/nnet3/decode_biglm.sh [options] <graph-dir> <old-LM-fst> <new-LM-fst> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/nnet3/decode_biglm.sh exp/chain/tree/graph data/test_dev93 exp/chain/tdnn/decode_dev93_tgpr"
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

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

utils/lang/check_phones_compatible.sh {$srcdir,$graphdir}/phones.txt || exit 1

for f in $graphdir/HCLG.fst $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
if [ -f $srcdir/cmvn_opts ]; then
    cmvn_opts=`cat $srcdir/cmvn_opts`
else
    cmvn_opts="--norm-means=false --norm-vars=false"
fi

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

## Set up features.
if [ -f $srcdir/online_cmvn ]; then online_cmvn=true
else online_cmvn=false; fi

if ! $online_cmvn; then
  echo "$0: feature type is raw"
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
else
  echo "$0: feature type is raw (apply-cmvn-online)"
  feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
fi

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
elif [ -f $srcdir/init/info.txt ]; then
    frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$srcdir/init/info.txt)
    if [ ! -z $frame_subsampling_factor ]; then
        frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
    fi
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
 nnet3-latgen-biglm-faster \
     $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt \
     $srcdir/final.mdl $graphdir/HCLG.fst "$oldlm_cmd" "$newlm_cmd" "$feats" \
     "$lat_wspecifier" || exit 1;

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
