#!/bin/bash

. ./cmd.sh
set -e 
stage=1
train_stage=-10
use_gpu=true
# splice_indexes="layer0/-4:-3:-2:-1:0:1:2:3:4 layer2/-5:-3:3"
#splice_indexes="layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3 layer3/-7:2"
splice_indexes="layer0/-2:-1:0:1:2 layer1/-4:-1:2 layer3/-3:3 layer4/-7:2"
common_egs_dir=
dir=exp/nnet2_online/nnet_ms_a

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
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi


# Run the common stages of training, including training the iVector extractor
local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 6 ]; then
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs 5 --num-jobs-initial 3 --num-jobs-final 18 \
    --num-hidden-layers 6 --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_hires_nodup2 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --add-layers-period 1 \
    --mix-up 4000 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 3000 \
    --pnorm-output-dim 300 \
    data/train_hires_nodup data/lang exp/tri4b_ali_nodup $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  for data in eval2000_hires train_hires_dev; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data} exp/nnet2_online/extractor exp/nnet2_online/ivectors_${data} || exit 1;
  done
fi

if [ $stage -le 8 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding (the one with --per-utt true)
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    # use already-built graphs.
    for data in eval2000_hires train_hires_dev; do
      steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" \
        --config conf/decode.config \
        --online-ivector-dir exp/nnet2_online/ivectors_${data} \
         $graph_dir data/${data} $dir/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi

if [ $stage -le 9 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
      data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 10 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000_hires train_hires_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config \
      --cmd "$decode_cmd" --nj 30 \
      "$graph_dir" data/${data} \
      ${dir}_online/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi

if [ $stage -le 11 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000_hires train_hires_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
        --per-utt true \
        "$graph_dir" data/${data} ${dir}_online/decode_${data}_sw1_${lm_suffix}_per_utt || exit 1;
    done
  done
fi

exit 0;
