#!/usr/bin/env bash

. ./cmd.sh


stage=1
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_a


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
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi


# stages 1 through 3 run in run_nnet2_common.sh.

local/online/run_nnet2_common.sh --stage  $stage || exit 1;


if [ $stage -le 4 ]; then
  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs 25 \
    --add-layers-period 1 \
    --num-hidden-layers 2 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 1000 \
    --pnorm-output-dim 200 \
    data/train data/lang exp/tri3b_ali $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    data/test exp/nnet2_online/extractor exp/nnet2_online/ivectors_test || exit 1;
fi


if [ $stage -le 6 ]; then
  # Note: comparing the results of this with run_online_decoding_nnet2_baseline.sh,
  # it's a bit worse, meaning the iVectors seem to hurt at this amount of data.
  # However, experiments by Haihua Xu (not checked in yet) on WSJ, show it helping
  # nicely.  This setup seems to have too little data for it to work, but it suffices
  # to demonstrate the scripts.   We will likely modify it to add noise to the
  # iVectors in training, which will tend to mitigate the over-training.
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    exp/tri3b/graph data/test $dir/decode  &

  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    exp/tri3b/graph_ug data/test $dir/decode_ug || exit 1;

  wait
fi

if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_per_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug_per_utt || exit 1;
  wait
fi

exit 0;


# the experiment (with GPU)
#for x in exp/nnet2_online/nnet_a/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
%WER 2.20 [ 276 / 12533, 37 ins, 61 del, 178 sub ] exp/nnet2_online/nnet_a/decode/wer_5
%WER 10.22 [ 1281 / 12533, 143 ins, 193 del, 945 sub ] exp/nnet2_online/nnet_a/decode_ug/wer_10

# This is the baseline with spliced non-CMVN cepstra and no iVector input.
# The difference is pretty small on RM; I expect it to be more clear-cut on larger corpora.
%WER 2.30 [ 288 / 12533, 35 ins, 57 del, 196 sub ] exp/nnet2_online/nnet_gpu_baseline/decode/wer_5
%WER 10.98 [ 1376 / 12533, 121 ins, 227 del, 1028 sub ] exp/nnet2_online/nnet_gpu_baseline/decode_ug/wer_10
 # and this is the same (baseline) using truly-online decoding; it probably only differs because
 # of slight decoding-parameter differences.
 %WER 2.31 [ 290 / 12533, 34 ins, 57 del, 199 sub ] exp/nnet2_online/nnet_gpu_baseline_online/decode/wer_5
 %WER 10.93 [ 1370 / 12533, 142 ins, 202 del, 1026 sub ] exp/nnet2_online/nnet_gpu_baseline_online/decode_ug/wer_9


# This is the online decoding.
# This truly-online per-utterance decoding gives essentially the same WER as the offline decoding, which is
# as we expect as the features and decoding parameters are the same.
# for x in exp/nnet2_online/nnet_gpu_online/decode*utt; do grep WER $x/wer_* | utils/best_wer.sh; done
%WER 2.28 [ 286 / 12533, 66 ins, 39 del, 181 sub ] exp/nnet2_online/nnet_a_online/decode_per_utt/wer_2
%WER 10.45 [ 1310 / 12533, 106 ins, 241 del, 963 sub ] exp/nnet2_online/nnet_a_online/decode_ug_per_utt/wer_12

# The following are online decoding, as above, but using previous utterances of
# the same speaker to refine the adaptation state.  It doesn't make much difference.
# for x in exp/nnet2_online/nnet_gpu_online/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done | grep -v utt
%WER 2.27 [ 285 / 12533, 42 ins, 62 del, 181 sub ] exp/nnet2_online/nnet_a_online/decode/wer_5
%WER 10.26 [ 1286 / 12533, 140 ins, 188 del, 958 sub ] exp/nnet2_online/nnet_a_online/decode_ug/wer_10


