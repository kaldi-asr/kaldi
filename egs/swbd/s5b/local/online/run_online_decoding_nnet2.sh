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
  # the _a is in case I want to change the parameters.
  dir=exp/nnet2_online/nnet_a_gpu 
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet_a
fi


if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/train_30k_nodup 512 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_100k_nodup exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_nodup exp/nnet2_online/extractor exp/nnet2_online/ivectors_train_nodup || exit 1;
fi


if [ $stage -le 4 ]; then
  # Because we have a lot of data here and we don't want the training to take
  # too long so we reduce the number of epochs from the defaults (15 + 5) to (5
  # + 2), and the (initial,final) learning rate from the defaults (0.04, 0.004)
  # to (0.01, 0.001). 
  # decided to let others run their jobs too (we only have 10 GPUs on our queue
  # at JHU).  The number of parameters is smaller than the baseline system we had in
  # mind (../nnet2/run_5d_gpu.sh, which had pnorm input/output dim 3000/300 and
  # 5 hidden layers, versus our 3000/300 and 5 hidden layers, even though we're
  # training on more data than the baseline system.  The motivation here is that we
  # want to demonstrate the capability of doing real-time decoding, and if the
  # network was too bug we wouldn't be able to decode in real-time using a CPU.
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --num-epochs 5 --num-epochs-extra 2 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_nodup \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 3000 \
    --pnorm-output-dim 300 \
    data/train_nodup data/lang exp/tri4b_ali_nodup $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  for data in eval2000 train_dev; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data} exp/nnet2_online/extractor exp/nnet2_online/ivectors_${data} || exit 1;
  done
fi


if [ $stage -le 6 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    # use already-built graphs.
    for data in eval2000 train_dev; do
      steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
          --online-ivector-dir exp/nnet2_online/ivectors_${data} \
         $graph_dir data/${data} $dir/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi


if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000 train_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
        "$graph_dir" data/${data} ${dir}_online/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi

[ $stage -eq 8 ] && exit 1;  ## This is temporary.

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000 train_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
        --per-utt true \
        "$graph_dir" data/${data} ${dir}_online/decode_${data}_sw1_${lm_suffix}_per_utt || exit 1;
    done
  done
fi


exit 0;



# get results on Dev with this command:
for x in exp/nnet2_online/nnet_a_gpu/decode_train_dev_sw1_*; do grep WER $x/wer_* | utils/best_wer.sh; done
# and results on eval2000 with this command:
for x in exp/nnet2_online/nnet_a_gpu/decode_eval2000_*; do grep Sum $x/score_*/*sys  | utils/best_wer.sh; done

# for a baseline (although not the very best baseline we could do), here is a GMM-based
# system trained on all the training data.
%WER 29.10 [ 14382 / 49427, 1963 ins, 3394 del, 9025 sub ] exp/tri4b/decode_train_dev_sw1_fsh_tgpr/wer_15
%WER 29.53 [ 14598 / 49427, 1885 ins, 3538 del, 9175 sub ] exp/tri4b/decode_train_dev_sw1_tg/wer_16
%WER 21.8 | 1831 21395 | 80.5 13.7 5.8 2.3 21.8 59.3 | exp/tri4b/decode_eval2000_sw1_fsh_tgpr/score_15/eval2000.ctm.swbd.filt.sys
%WER 22.4 | 1831 21395 | 80.0 13.9 6.1 2.4 22.4 60.0 | exp/tri4b/decode_eval2000_sw1_tg/score_16/eval2000.ctm.swbd.filt.sys



# our neural net trained with iVector input, tested in batch mode.
%WER 22.32 [ 11033 / 49427, 1198 ins, 3121 del, 6714 sub ] exp/nnet2_online/nnet_a_gpu/decode_train_dev_sw1_tg/wer_12
%WER 21.75 [ 10751 / 49427, 1176 ins, 3135 del, 6440 sub ] exp/nnet2_online/nnet_a_gpu/decode_train_dev_sw1_fsh_tgpr/wer_12
%WER 18.1 | 1831 21395 | 83.7 10.8 5.5 1.8 18.1 56.3 | exp/nnet2_online/nnet_a_gpu/decode_eval2000_sw1_fsh_tgpr/score_12/eval2000.ctm.swbd.filt.sys
%WER 18.5 | 1831 21395 | 83.4 11.2 5.4 1.9 18.5 57.2 | exp/nnet2_online/nnet_a_gpu/decode_eval2000_sw1_tg/score_12/eval2000.ctm.swbd.filt.sys


# the same thing tested using truly-online decoding, tested separately per
# utterance (which should in principle give the same results as the batch-mode
# test, which also was per-utterance); I'm not sure what the reason for the slight improvement 
# is.
%WER 22.05 [ 10897 / 49427, 1355 ins, 2849 del, 6693 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_train_dev_sw1_tg_per_utt/wer_11
%WER 21.53 [ 10641 / 49427, 1316 ins, 2848 del, 6477 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_train_dev_sw1_fsh_tgpr_per_utt/wer_11


# truly-online decoding, but thuis time carrying forward the adaptation state (the iVector
# and associated CMVN) from one utterance to the next within the same speaker.
%WER 21.43 [ 10593 / 49427, 1231 ins, 2827 del, 6535 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_train_dev_sw1_tg/wer_11
%WER 21.23 [ 10494 / 49427, 1240 ins, 2827 del, 6427 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_train_dev_sw1_fsh_tgpr/wer_11

