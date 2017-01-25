#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -e
set -o pipefail 
set -u

# general opts
iter=final
stage=0
decode_num_jobs=30
num_jobs=30
affix=

sad_iter=final

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

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <data-dir> <sad-nnet-dir> <lang-dir> <graph-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)   # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 dev_aspire data/lang exp/tri5a/graph_pp exp/nnet3/tdnn"
  exit 1;
fi

data_set=$1 
sad_nnet_dir=$2
lang=$3 # data/lang
graph=$4 #exp/tri5a/graph_pp
dir=$5 # exp/nnet3/tdnn

model_affix=`basename $dir`
ivector_dir=exp/nnet3
ivector_affix=${affix:+_$affix}_chain_${model_affix}_iter$iter
affix=_${affix}_iter${iter}
act_data_set=${data_set} # we will modify the data dir, when segmenting it
                         # so we will keep track of original data dirfor the glm and stm files

if [[ "$data_set" =~ "test_aspire" ]]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
elif [[ "$data_set" =~ "eval_aspire" ]]; then
  out_file=single_eval${affix}_$model_affix.ctm
elif [[ "$data_set" =~  "dev_aspire" ]]; then
  # we will just decode the directory without oracle segments file
  # as we would like to operate in the actual evaluation condition
  out_file=single_dev${affix}_${model_affix}.ctm
else 
  exit 1
fi

if [ $stage -le 1 ]; then
  steps/segmentation/do_segmentation_data_dir.sh --reco-nj $num_jobs \
    --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp --iter $sad_iter \
    --do-downsampling false --extra-left-context 100 --extra-right-context 20 \
    --output-name output-speech --frame-subsampling-factor 6 \
    data/${data_set} $sad_nnet_dir mfcc_hires_bp data/${data_set}${affix}
  # Output will be in data/${data_set}_seg
fi

# uniform segmentation script would have created this dataset
# so update that script if you plan to change this variable
segmented_data_set=${data_set}${affix}_seg

if [ $stage -le 2 ]; then
  mfccdir=mfcc_reverb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/aspire-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh data/${segmented_data_set} data/${segmented_data_set}_hires
  steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf data/${segmented_data_set}_hires \
    exp/make_reverb_hires/${segmented_data_set} $mfccdir
  steps/compute_cmvn_stats.sh data/${segmented_data_set}_hires \
    exp/make_reverb_hires/${segmented_data_set} $mfccdir
  utils/fix_data_dir.sh data/${segmented_data_set}_hires
  utils/validate_data_dir.sh --no-text data/${segmented_data_set}_hires
fi

decode_dir=$dir/decode_${segmented_data_set}_pp
if [ $stage -le 5 ]; then
  echo "Extracting i-vectors, stage 2"
  # this does offline decoding, except we estimate the iVectors per
  # speaker, excluding silence (based on alignments from a DNN decoding), with a
  # different script.  This is just to demonstrate that script.
  # the --sub-speaker-frames is optional; if provided, it will divide each speaker
  # up into "sub-speakers" of at least that many frames... can be useful if
  # acoustic conditions drift over time within the speaker's data.
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
    --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
    data/${segmented_data_set}_hires $lang $ivector_dir/extractor \
    $ivector_dir/ivectors_${segmented_data_set}${ivector_affix};
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
    --resolve-overlaps false \
    --tune-hyper true \
    $lang $decode_dir $act_data_set $segmented_data_set $out_file
fi

# Two-pass decoding baseline
# %WER 27.8 | 2120 27217 | 78.2 13.6 8.2 6.0 27.8 75.9 | -0.613 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iterfinal_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# Using automatic segmentation 
# %WER 28.2 | 2120 27214 | 76.5 12.4 11.1 4.7 28.2 75.2 | -0.522 | exp/chain/tdnn_7b/decode_dev_aspire_seg_v7_n_stddev_iterfinal_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
