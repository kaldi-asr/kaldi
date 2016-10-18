#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -e

# general opts
iter=final
stage=0
decode_num_jobs=30
num_jobs=30
affix=

# segmentation opts
window=10
overlap=5

# ivector opts
max_count=75 # parameter for extract_ivectors.sh
sub_speaker_frames=6000
ivector_scale=0.75
filter_ctm=true
weights_file=
silence_weight=0.00001

# decode opts
pass2_decode_opts="--min-active 1000"
lattice_beam=8
extra_left_context=0 # change for (B)LSTM
extra_right_context=0 # change for BLSTM
frames_per_chunk=50 # change for (B)LSTM
acwt=0.1 # important to change this when using chain models
post_decode_acwt=1.0 # important to change this when using chain models

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

data_set=$1 #select from {dev_aspire, test_aspire, eval_aspire}
lang=$2 # data/lang
graph=$3 #exp/tri5a/graph_pp
dir=$4 # exp/nnet3/tdnn

model_affix=`basename $dir`
ivector_dir=exp/nnet3
ivector_affix=${affix:+_$affix}_chain_${model_affix}_iter$iter
affix=_${affix}_iter${iter}
act_data_set=${data_set} # we will modify the data set, when uniformly segmenting it
                         # so we will keep track of original data set for the glm and stm files

if [ $stage -le 1 ]; then
  local/generate_uniformly_segmented_data_dir.sh  \
    --overlap $overlap --window $window $data_set
fi

if [ "$data_set" == "test_aspire" ]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
elif [ "$data_set" == "eval_aspire" ]; then
  out_file=single_eval${affix}_$model_affix.ctm
elif [ "$data_set" ==  "dev_aspire" ]; then
  # we will just decode the directory without oracle segments file
  # as we would like to operate in the actual evaluation condition
  data_set=${data_set}_whole
  out_file=single_dev${affix}_${model_affix}.ctm
fi

# uniform segmentation script would have created this dataset
# so update that script if you plan to change this variable
segmented_data_set=${data_set}_uniformsegmented_win${window}_over${overlap}

if [ $stage -le 2 ]; then
  echo "Extracting i-vectors, stage 1"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    --max-count $max_count \
    data/${segmented_data_set}_hires $ivector_dir/extractor \
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

decode_dir=$dir/decode_${segmented_data_set}${affix}_pp
# generate the lattices
if [ $stage -le 3 ]; then
  echo "Generating lattices, stage 1"
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config \
    --acwt $acwt --post-decode-acwt $post_decode_acwt \
    --extra-left-context $extra_left_context  \
    --extra-right-context $extra_right_context  \
    --frames-per-chunk "$frames_per_chunk" \
    --online-ivector-dir $ivector_dir/ivectors_${segmented_data_set}${ivector_affix}${ivector_scale_affix}_stage1 \
    --skip-scoring true --iter $iter \
    $graph data/${segmented_data_set}_hires ${decode_dir}_stage1;
fi

if [ $stage -le 4 ]; then
  if $filter_ctm; then
    if [ ! -z $weights_file ]; then
      echo "$0: Using provided vad weights file $weights_file"
      ivector_extractor_input=$weights_file
    else
      echo "$0 : Generating vad weights file"
      ivector_extractor_input=${decode_dir}_stage1/weights${affix}.gz
      local/extract_vad_weights.sh --cmd "$decode_cmd" --iter $iter \
        data/${segmented_data_set}_hires $lang \
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
    data/${segmented_data_set}_hires $lang $ivector_dir/extractor \
    $ivector_extractor_input $ivector_dir/ivectors_${segmented_data_set}${ivector_affix};
fi

if [ $stage -le 6 ]; then
  echo "Generating lattices, stage 2 with --acwt $acwt"
  rm -f ${decode_dir}_tg/.error
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config $pass2_decode_opts \
      --acwt $acwt --post-decode-acwt $post_decode_acwt \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --frames-per-chunk "$frames_per_chunk" \
      --skip-scoring true --iter $iter --lattice-beam $lattice_beam \
      --online-ivector-dir $ivector_dir/ivectors_${segmented_data_set}${ivector_affix} \
     $graph data/${segmented_data_set}_hires ${decode_dir}_tg || touch ${decode_dir}_tg/.error
  [ -f ${decode_dir}_tg/.error ] && echo "$0: Error decoding" && exit 1;
fi

if [ $stage -le 7 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_set}_hires \
    ${decode_dir}_{tg,fg};
fi

decode_dir=${decode_dir}_fg

if [ $stage -le 8 ]; then
  local/score_aspire.sh --cmd "$decode_cmd" \
    --min-lmwt 1 --max-lmwt 20 \
    --word-ins-penalties "0.0,0.25,0.5,0.75,1.0" \
    --ctm-beam 6 \
    --iter $iter \
    --decode-mbr true \
    --window $window \
    --overlap $overlap \
    --tune-hyper true \
    $lang $decode_dir $act_data_set $segmented_data_set $out_file
fi
