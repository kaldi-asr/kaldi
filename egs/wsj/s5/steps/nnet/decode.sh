#!/usr/bin/env bash

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely), Daniel Povey
# Apache 2.0

# Begin configuration section.
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

blocksoftmax_dims=   # 'csl' with block-softmax dimensions: dim1,dim2,dim3,...
blocksoftmax_active= # '1' for the 1st block,

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=8.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
nnet_forward_opts="--no-softmax=true --prior-scale=1.0"

skip_scoring=false
scoring_opts="--min-lmwt 4 --max-lmwt 15"

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts=   # Ignored now.
use_gpu="no" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the DNN and transition model is."
   echo "e.g.: $0 exp/dnn1/graph_tgpr data/test exp/dnn1/decode_tgpr"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of srcdir."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --srcdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   echo "  --acwt <float>                                   # select acoustic scale for decoding"
   echo "  --scoring-opts <opts>                            # options forwarded to local/score.sh"
   echo "  --num-threads <N>                                # N>1: run multi-threaded decoder"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
[ -z $srcdir ] && srcdir=`dirname $dir`; # Default model directory one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$model" ] && model=$srcdir/final.mdl
[ -z "$feature_transform" -a -e $srcdir/final.feature_transform ] && feature_transform=$srcdir/final.feature_transform
#
[ -z "$class_frame_counts" -a -f $srcdir/prior_counts ] && class_frame_counts=$srcdir/prior_counts # priority,
[ -z "$class_frame_counts" ] && class_frame_counts=$srcdir/ali_train_pdf.counts

# Check that files exist,
for f in $sdata/1/feats.scp $nnet $model $feature_transform $class_frame_counts $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"


# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
online_cmvn_opts=
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/online_cmvn_opts ] && online_cmvn_opts=$(cat $D/online_cmvn_opts)
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn-online (optional),
[ -n "$online_cmvn_opts" -a ! -f $D/global_cmvn_stats.mat ] && echo "$0: Missing $D/global_cmvn_stats.mat" && exit 1
[ -n "$online_cmvn_opts" ] && feats="$feats apply-cmvn-online $online_cmvn_opts --spk2utt=ark:$srcdata/spk2utt $D/global_cmvn_stats.mat ark:- ark:- |"
# apply-cmvn (optional),
[ -n "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ -n "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ -n "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  [ -z $ivector ] && echo "Missing --ivector, they were used in training!" && exit 1
  # Get the tool,
  ivector_append_tool=append-vector-to-feats # default,
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  # Check dims,
  feats_job_1=$(sed 's:JOB:1:g' <(echo $feats))
  dim_raw=$(feat-to-dim "$feats_job_1" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats_job_1 $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  [ $dim_ivec != "$(cat $D/ivector_dim)" ] && \
    echo "Error, i-vector dim. mismatch (expected $(cat $D/ivector_dim), got $dim_ivec in '$ivector')" && \
    exit 1
  # Append to feats,
  feats="$feats $ivector_append_tool ark:- '$ivector' ark:- |"
fi

# select a block from blocksoftmax,
if [ ! -z "$blocksoftmax_dims" ]; then
  # blocksoftmax_active is a csl! dim1,dim2,dim3,...
  [ -z "$blocksoftmax_active" ] && echo "$0 Missing option --blocksoftmax-active N" && exit 1
  # getting dims,
  dim_total=$(awk -F'[:,]' '{ for(i=1;i<=NF;i++) { sum += $i }; print sum; }' <(echo $blocksoftmax_dims))
  dim_block=$(awk -F'[:,]' -v active=$blocksoftmax_active '{ print $active; }' <(echo $blocksoftmax_dims))
  offset=$(awk -F'[:,]' -v active=$blocksoftmax_active '{ sum=0; for(i=1;i<active;i++) { sum += $i }; print sum; }' <(echo $blocksoftmax_dims))
  # create components which select a block,
  nnet-initialize <(echo "<Copy> <InputDim> $dim_total <OutputDim> $dim_block <BuildVector> $((1+offset)):$((offset+dim_block)) </BuildVector>";
                    echo "<Softmax> <InputDim> $dim_block <OutputDim> $dim_block") $dir/copy_and_softmax.nnet
  # nnet is assembled on-the fly, <BlockSoftmax> is removed, while <Copy> + <Softmax> is added,
  nnet="nnet-concat 'nnet-copy --remove-last-components=1 $nnet - |' $dir/copy_and_softmax.nnet - |"
fi

# Run the decoding in the queue,
if [ $stage -le 0 ]; then
  $cmd --num-threads $((num_threads+1)) JOB=1:$nj $dir/log/decode.JOB.log \
    nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu "$nnet" "$feats" ark:- \| \
    latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

# Run the scoring
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
fi

exit 0;
