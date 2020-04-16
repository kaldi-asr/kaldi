#!/usr/bin/env bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script does online decoding, unlike local/nnet3/decode.sh which does 2-pass decoding with
# uniform segments.

set -e

# general opts
iter=
stage=0
decode_num_jobs=30
num_jobs=30
affix=

# segmentation opts
window=10
overlap=5

# ivector+decode opts
# tuned based on the ASpIRE nnet2 online system
max_count=75
max_state_duration=40
silence_weight=0.00001

pass2_decode_opts="--min-active 1000"
lattice_beam=8
extra_left_context=0 # change for (B)LSTM
extra_right_context=0 # change for BLSTM
frames_per_chunk=50 # change for (B)LSTM
acwt=0.1 # important to change this when using chain models
post_decode_acwt=1.0 # important to change this when using chain models
extra_left_context_initial=0

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

data_set=$1 #select from {dev_aspire, test_aspire, eval_aspire}
lang=$2 # data/lang
graph=$3 #exp/tri5a/graph_pp
dir=$4 # exp/nnet3/tdnn

model_affix=`basename $dir`
affix=_${affix}${iter:+_iter${iter}}

segmented_data_set=${data_set}_uniformsegmented
if [ $stage -le 1 ]; then
  local/generate_uniformly_segmented_data_dir.sh  \
    --overlap $overlap --window $window $data_set $segmented_data_set
fi

if [[ "$data_set" =~ "test_aspire" ]]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
  act_data_set=test_aspire
elif [[ "$data_set" =~ "eval_aspire" ]]; then
  out_file=single_eval${affix}_$model_affix.ctm
  act_data_set=eval_aspire
elif [[ "$data_set" =~  "dev_aspire" ]]; then
  # we will just decode the directory without oracle segments file
  # as we would like to operate in the actual evaluation condition
  out_file=single_dev${affix}_${model_affix}.ctm
  act_data_set=dev_aspire
else
  echo "$0: Unknown data-set $data_set"
  exit 1
fi

if [ $stage -le 2 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    --max-count $max_count \
      $lang exp/nnet3/extractor "$dir" ${dir}_online
fi

decode_dir=${dir}_online/decode_${segmented_data_set}${affix}_pp
if [ $stage -le 3 ]; then
  echo "Generating lattices, with --acwt $acwt and --post-decode-acwt $post_decode_acwt "
      # the following options have not yet been implemented
      # --frames-per-chunk "$frames_per_chunk"
      #--extra-left-context $extra_left_context  \
      #--extra-right-context $extra_right_context  \
  steps/online/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" \
      --config conf/decode.config $pass2_decode_opts \
      --acwt $acwt --post-decode-acwt $post_decode_acwt \
      --extra-left-context-initial $extra_left_context_initial \
      --silence-weight $silence_weight \
      --per-utt true \
      --skip-scoring true ${iter:+--iter $iter} --lattice-beam $lattice_beam \
     $graph data/${segmented_data_set}_hires ${decode_dir}_tg || \
     { echo "$0: Error decoding" && exit 1; }
fi

if [ $stage -le 4 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_set}_hires \
    ${decode_dir}_{tg,fg};
fi

decode_dir=${decode_dir}_fg
if [ $stage -le 5 ]; then
  local/score_aspire.sh --cmd "$decode_cmd" \
    $score_opts \
    --word-ins-penalties "0.0,0.25,0.5,0.75,1.0" \
    --ctm-beam 6 \
    ${iter:+--iter $iter} \
    --decode-mbr true \
    --tune-hyper true \
    $lang $decode_dir $act_data_set $segmented_data_set $out_file
fi
