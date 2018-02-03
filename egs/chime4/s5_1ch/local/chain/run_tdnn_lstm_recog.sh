#!/bin/bash

set -e -o pipefail

stage=0
nj=30
train=noisy
enhan=$1
mdir=$2
train_set=tr05_multi_${train}
test_sets="dt05_real_$enhan dt05_simu_$enhan et05_real_$enhan et05_simu_$enhan"
gmm=tri3b_tr05_multi_${train} # this is the source gmm-dir that we'll use for alignments; it
                              # should have alignments for the specified training data.
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <model dir>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies acoustic and language model directory"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_multi_${train} ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check ivector extractor
if [ ! -d $mdir/exp/nnet3${nnet3_affix}/extractor ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/nnet3${nnet3_affix}/extractor ]; then
  echo "copy $mdir/exp/nnet3${nnet3_affix}/extractor"
  mkdir -p exp/nnet3${nnet3_affix}
  cp -r $mdir/exp/nnet3${nnet3_affix}/extractor exp/nnet3${nnet3_affix}/
fi

# check tdnn-lstm graph
if [ ! -d $mdir/exp/chain${nnet3_affix}/tree_a_sp/graph_tgpr_5k ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/chain${nnet3_affix}/tree_a_sp/graph_tgpr_5k ]; then
  echo "copy $mdir/exp/chain${nnet3_affix}/tree_a_sp/graph_tgpr_5k"
  mkdir -p exp/chain${nnet3_affix}/tree_a_sp
  cp -r $mdir/exp/chain${nnet3_affix}/tree_a_sp/graph_tgpr_5k exp/chain${nnet3_affix}/tree_a_sp/
fi

# check dir
if [ ! -d $mdir/exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp ]; then
  echo "error, set $mdir correctly"
  exit 1;
elif [ ! -d exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp ]; then
  echo "copy $mdir/exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp"
  cp -r $mdir/exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp exp/chain${nnet3_affix}/
  rm -rf exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp/decode_*
  rm -rf exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp/best_*
fi

dir=exp/chain${nnet3_affix}/tdnn_lstm${affix}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=$mdir/exp/chain${nnet3_affix}/tree_a_sp

# make ivector for dev and eval
if [ $stage -le 2 ]; then
  for datadir in ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  
  # extracting hires features
  for datadir in ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  
  # extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in ${test_sets}; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
    data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
    exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

if [ $stage -le 18 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr_5k; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 19 ]; then
  # 'looped' decoding.
  # note: you should NOT do this decoding step for setups that have bidirectional
  # recurrence, like BLSTMs-- it doesn't make sense and will give bad results.
  # we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr_5k; do
        steps/nnet3/decode_looped.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk 30 \
          --nj $nspk --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_looped_${lmtype}_${data_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if $test_online_decoding && [ $stage -le 20 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      for lmtype in tgpr; do
        steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $tree_dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# scoring
if [ $stage -le 20 ]; then
  # decoded results of enhanced speech using TDNN AMs trained with enhanced data
  local/chime4_calc_wers.sh exp/chain/tdnn_lstm${affix}_sp $enhan exp/chain/tree_a_sp/graph_tgpr_5k \
    > exp/chain/tdnn_lstm${affix}_sp/best_wer_$enhan.result
  head -n 15 exp/chain/tdnn_lstm${affix}_sp/best_wer_$enhan.result
  
  echo "score looped decoding results"
  local/chime4_calc_wers_looped.sh exp/chain/tdnn_lstm${affix}_sp $enhan exp/chain/tree_a_sp/graph_tgpr_5k \
    > exp/chain/tdnn_lstm${affix}_sp/best_wer_looped_$enhan.result
  head -n 15 exp/chain/tdnn_lstm${affix}_sp/best_wer_looped_$enhan.result
fi

exit 0;
