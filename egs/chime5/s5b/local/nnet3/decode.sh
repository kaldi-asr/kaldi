#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -e

# general opts
iter=
stage=0
decode_num_jobs=30
num_jobs=30
affix=

## segmentation opts
#window=10
#overlap=5

# ivector opts
max_count=75 # parameter for extract_ivectors.sh
sub_speaker_frames=6000
ivector_scale=0.75
filter_ctm=true
weights_file=
silence_weight=0.00001
ivector_dir=exp/nnet3

# decode opts
pass2_decode_opts="--min-active 1000"
lattice_beam=8
extra_left_context=0 # change for (B)LSTM
extra_right_context=0 # change for BLSTM
frames_per_chunk=50 # change for (B)LSTM
acwt=0.1 # important to change this when using chain models
post_decode_acwt=1.0 # important to change this when using chain models
extra_left_context_initial=0
extra_right_context_final=0

score_opts="--min-lmwt 6 --max-lmwt 13"

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <graph-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)   # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 dev_aspire data/lang exp/tri5a/graph_pp exp/nnet3/tdnn"
  exit 1;
fi

data=$1 #select from {dev_aspire, test_aspire, eval_aspire}
lang=$2 # data/lang
graph=$3 #exp/tri5a/graph_pp
dir=$4 # exp/nnet3/tdnn

model_affix=`basename $dir`
ivector_affix=${affix:+_$affix}_chain_${model_affix}${iter:+_iter$iter}
affix=${affix:+_${affix}}${iter:+_iter${iter}}

segmented_data=${data}
if [ $stage -le 1 ]; then
  if [ ! -s ${data}_hires/feats.scp ]; then
    utils/copy_data_dir.sh $data ${data}_hires
    steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 20 --cmd "$train_cmd" ${data}_hires
    steps/compute_cmvn_stats.sh ${data}_hires
    utils/fix_data_dir.sh ${data}_hires
  fi
fi

#if [ $stage -le 1 ]; then
#  local/generate_uniformly_segmented_data_dir.sh  \
#    --overlap $overlap --window $window $data $segmented
#fi

segmented_data_set=$(basename $segmented_data)
if [ $stage -le 2 ]; then
  echo "Extracting i-vectors, stage 1"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    --max-count $max_count \
    ${segmented_data}_hires $ivector_dir/extractor \
    $ivector_dir/ivectors_${segmented_data_set}${ivector_affix}_stage1;
  # float comparisons are hard in bash
  if [ `bc <<< "$ivector_scale != 1"` -eq 1 ]; then
    ivector_scale_affix=_scale$ivector_scale
  else
    ivector_scale_affix=
  fi

  if [ ! -z "$ivector_scale_affix" ]; then
    echo "$0: Scaling iVectors, stage 1"
    srcdir=$ivector_dir/ivectors_${segmented_data_set}${ivector_affix}_stage1
    outdir=$ivector_dir/ivectors_${segmented_data_set}${ivector_affix}${ivector_scale_affix}_stage1
    mkdir -p $outdir
    copy-matrix --scale=$ivector_scale scp:$srcdir/ivector_online.scp ark:- | \
      copy-feats --compress=true ark:-  ark,scp:$outdir/ivector_online.ark,$outdir/ivector_online.scp;
    cp $srcdir/ivector_period $outdir/ivector_period
  fi
fi

decode_dir=$dir/decode_${segmented_data_set}${affix}
# generate the lattices
if [ $stage -le 3 ]; then
  echo "Generating lattices, stage 1"
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" \
    --acwt $acwt --post-decode-acwt $post_decode_acwt \
    --extra-left-context $extra_left_context  \
    --extra-right-context $extra_right_context  \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk "$frames_per_chunk" \
    --online-ivector-dir $ivector_dir/ivectors_${segmented_data_set}${ivector_affix}${ivector_scale_affix}_stage1 \
    --skip-scoring true ${iter:+--iter $iter} \
    $graph ${segmented_data}_hires ${decode_dir}_stage1;
fi

if [ $stage -le 4 ]; then
  if $filter_ctm; then
    if [ ! -z $weights_file ]; then
      echo "$0: Using provided vad weights file $weights_file"
      ivector_extractor_input=$weights_file
    else
      echo "$0 : Generating vad weights file"
      ivector_extractor_input=${decode_dir}_stage1/weights${affix}.gz
      local/extract_vad_weights.sh --cmd "$decode_cmd" ${iter:+--iter $iter} \
        ${segmented_data}_hires $lang \
        ${decode_dir}_stage1 $ivector_extractor_input
    fi
  else
    # just use all the frames
    ivector_extractor_input=${decode_dir}_stage1
  fi
fi

if [ $stage -le 5 ]; then
  echo "Extracting i-vectors, stage 2 with input $ivector_extractor_input"
  # this does offline decoding, except we estimate the iVectors per
  # speaker, excluding silence (based on alignments from a DNN decoding), with a
  # different script.  This is just to demonstrate that script.
  # the --sub-speaker-frames is optional; if provided, it will divide each speaker
  # up into "sub-speakers" of at least that many frames... can be useful if
  # acoustic conditions drift over time within the speaker's data.
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
    --silence-weight $silence_weight \
    --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
    ${segmented_data}_hires $lang $ivector_dir/extractor \
    $ivector_extractor_input $ivector_dir/ivectors_${segmented_data_set}${ivector_affix};
fi

if [ $stage -le 6 ]; then
  echo "Generating lattices, stage 2 with --acwt $acwt"
  rm -f ${decode_dir}/.error
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" $pass2_decode_opts \
      --acwt $acwt --post-decode-acwt $post_decode_acwt \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --extra-left-context-initial $extra_left_context_initial \
      --extra-right-context-final $extra_right_context_final \
      --frames-per-chunk "$frames_per_chunk" \
      --skip-scoring false ${iter:+--iter $iter} --lattice-beam $lattice_beam \
      --online-ivector-dir $ivector_dir/ivectors_${segmented_data_set}${ivector_affix} \
     $graph ${segmented_data}_hires ${decode_dir} || touch ${decode_dir}/.error
  [ -f ${decode_dir}/.error ] && echo "$0: Error decoding" && exit 1;
fi
exit 0
