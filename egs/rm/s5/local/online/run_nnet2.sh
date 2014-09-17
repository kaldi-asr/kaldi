#!/bin/bash

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
. cmd.sh
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
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  dir=exp/nnet2_online/nnet_gpu
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet
fi


if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 --num-frames 200000 \
    data/train 256 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # use a smaller iVector dim (50) than the default (100) because RM has a very
  # small amount of data.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
    --ivector-dim 50 \
   data/train exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    --utts-per-spk-max 2 \
    data/train exp/nnet2_online/extractor exp/nnet2_online/ivectors || exit 1;
fi


if [ $stage -le 4 ]; then
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs-extra 10 --add-layers-period 1 \
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
#for x in exp/nnet2_online/nnet_gpu/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
%WER 2.27 [ 285 / 12533, 43 ins, 50 del, 192 sub ] exp/nnet2_online/nnet_gpu/decode/wer_4
%WER 10.40 [ 1303 / 12533, 133 ins, 200 del, 970 sub ] exp/nnet2_online/nnet_gpu/decode_ug/wer_11


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
%WER 2.21 [ 277 / 12533, 45 ins, 48 del, 184 sub ] exp/nnet2_online/nnet_gpu_online/decode_per_utt/wer_4
%WER 10.27 [ 1287 / 12533, 142 ins, 186 del, 959 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug_per_utt/wer_10

# The following are online decoding, as above, but using previous utterances of
# the same speaker to refine the adaptation state.  It doesn't make much difference.
# for x in exp/nnet2_online/nnet_gpu_online/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done | grep -v utt
%WER 2.20 [ 276 / 12533, 25 ins, 69 del, 182 sub ] exp/nnet2_online/nnet_gpu_online/decode/wer_8
%WER 10.14 [ 1271 / 12533, 127 ins, 198 del, 946 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug/wer_11
