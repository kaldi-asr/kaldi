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
    data/train 512 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
   data/train exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
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



# the experiment (no GPU)
#for x in exp/nnet2_online/nnet/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
%WER 2.75 [ 345 / 12533, 43 ins, 81 del, 221 sub ] exp/nnet2_online/nnet/decode/wer_7
%WER 10.94 [ 1371 / 12533, 133 ins, 220 del, 1018 sub ] exp/nnet2_online/nnet/decode_ug/wer_11

# for x in exp/nnet2_online/nnet_baseline/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
# This is the baseline with spliced non-CMVN cepstra and no iVector input.  This
# baseline was actually better than the experiment itself, due to severe
# overtraining in this RM setup (overtraining is something the Google paper
# noted too), but in larger setups it seems to actually help (Haihua Xu tested
# it on WSJ).  Soon I'll create larger-scale test setups, and look into
# mitigating the overtraining on RM.
%WER 2.30 [ 288 / 12533, 44 ins, 51 del, 193 sub ] exp/nnet2_online/nnet_baseline/decode/wer_4
%WER 10.70 [ 1341 / 12533, 122 ins, 221 del, 998 sub ] exp/nnet2_online/nnet_baseline/decode_ug/wer_10

# This is the online decoding.
#for x in exp/nnet2_online/nnet_online/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
# The per-utterance decoding gives essentially the same WER as the offline decoding, which is
# as we expect as the features and decoding parameters are the same.
%WER 2.73 [ 342 / 12533, 48 ins, 64 del, 230 sub ] exp/nnet2_online/nnet_online/decode_per_utt/wer_6
%WER 10.99 [ 1377 / 12533, 122 ins, 221 del, 1034 sub ] exp/nnet2_online/nnet_online/decode_ug_per_utt/wer_12

# The following are online decoding, as above, but using previous utterances of
# the same speaker to refine the adaptation state.  This setup is probably too
# small to really know whether this is helpful.  If it's not helpful to use
# previous utterances of the same speaker, it probably means we need to do
# matched training.
%WER 2.61 [ 327 / 12533, 41 ins, 73 del, 213 sub ] exp/nnet2_online/nnet_online/decode/wer_7
%WER 11.18 [ 1401 / 12533, 137 ins, 213 del, 1051 sub ] exp/nnet2_online/nnet_online/decode_ug/wer_12
