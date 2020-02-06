#!/usr/bin/env bash

# example script for online-nnet2 system training and decoding,
# based on the one for fisher-English.

# note: run_nnet2_ms.sh gives better results, but it's slower to train.

. ./cmd.sh


stage=0
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_a

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

local/online/run_nnet2_common.sh --stage $stage

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  # The size of the system is kept rather smaller than the run_7a_960.sh system:
  # this is because we want it to be small enough that we could plausibly run it
  # in real-time.
  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
    --num-epochs 8 --num-jobs-nnet 6 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_960_hires \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    --num-hidden-layers 4 \
    --mix-up 12000 \
    data/train_960_hires data/lang exp/tri6b $dir  || exit 1;
fi

if [ $stage -le 8 ]; then
  # dump iVectors for the testing data.
  for test in dev_clean dev_other; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${test}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_$test || exit 1;
  done
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  for test in dev_clean dev_other; do
    steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      --online-ivector-dir exp/nnet2_online/ivectors_${test} \
      exp/tri6b/graph_tgsmall data/${test}_hires $dir/decode_${test}_tgsmall || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/${test}_hires $dir/decode_${test}_{tgsmall,tgmed}  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test $dir/decode_${test}_{tgsmall,tglarge} || exit 1;
  done
fi


if [ $stage -le 10 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 11 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for test in test_clean test_other dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri6b/graph_tgsmall data/$test ${dir}_online/decode_${test}_tgsmall || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tglarge} || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,fglarge} || exit 1;
  done
fi

if [ $stage -le 12 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for test in dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      --per-utt true exp/tri6b/graph_tgsmall data/$test ${dir}_online/decode_${test}_tgsmall_utt || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}_utt  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tglarge}_utt || exit 1;
  done
fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector (--online false)
  for test in dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      --per-utt true --online false exp/tri6b/graph_tgsmall data/$test \
        ${dir}_online/decode_${test}_tgsmall_utt_offline || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}_utt_offline  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tglarge}_utt_offline || exit 1;
  done
fi

exit 0;
###### Comment out the "exit 0" above to run the multi-threaded decoding. #####

if [ $stage -le 14 ]; then
  # Demonstrate the multi-threaded decoding.
  test=dev_clean
  steps/online/nnet2/decode.sh --threaded true \
    --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
    --per-utt true exp/tri6b/graph_tgsmall data/$test \
    ${dir}_online/decode_${test}_tgsmall_utt_threaded || exit 1;
fi

if [ $stage -le 15 ]; then
  # Demonstrate the multi-threaded decoding with endpointing.
  test=dev_clean
  steps/online/nnet2/decode.sh --threaded true --do-endpointing true \
    --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
    --per-utt true exp/tri6b/graph_tgsmall data/$test \
    ${dir}_online/decode_${test}_tgsmall_utt_threaded_ep || exit 1;
fi

exit 0;
