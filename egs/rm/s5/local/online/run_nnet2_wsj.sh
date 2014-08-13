#!/bin/bash

# This script assumes you have previously run the WSJ example script including
# the optional part local/online/run_online_decoding_nnet2.sh.  It builds a
# neural net for online decoding on top of the network we previously trained on
# WSJ, by keeping everything but the last layer of that network and then
# training just the last layer on our data.

stage=0
set -e

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
  dir=exp/nnet2_online_wsj/nnet_gpu
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train_gpu
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_gpu_online
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online_wsj/nnet
  trainfeats=exp/nnet2_online_wsj/wsj_activations_train
  srcdir=../../wsj/s5/exp/nnet2_online/nnet_a_online
fi


if [ $stage -le 0 ]; then
  echo "$0: dumping activations from WSJ model"
  steps/online/nnet2/dump_nnet_activations.sh --cmd "$train_cmd" --nj 30 \
     data/train $srcdir $trainfeats
fi

if [ $stage -le 1 ]; then
  echo "$0: training 0-hidden-layer model on top of WSJ activations"
  steps/nnet2/retrain_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --cmd "$decode_cmd" \
    --num-jobs-nnet 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     $trainfeats/data data/lang exp/tri3b_ali $dir 
fi

if [ $stage -le 2 ]; then
  echo "$0: formatting combined model for online decoding."
  steps/online/nnet2/prepare_online_decoding_retrain.sh $srcdir $dir ${dir}_online
fi

# Note: at this point it might be possible to further train the combined model
# by doing backprop through all of it.  We haven't implemented this yet.

if [ $stage -le 3 ]; then
  # do online decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

if [ $stage -le 4 ]; then
  # do online per-utterance decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --per-utt true \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug_utt || exit 1;
  wait
fi


# Here are the results:
# grep WER exp/nnet2_online_wsj/nnet_gpu_online/decode/wer_* | utils/best_wer.sh 
#%WER 1.61 [ 202 / 12533, 22 ins, 46 del, 134 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode/wer_3
#a11:s5: grep WER exp/nnet2_online_wsj/nnet_gpu_online/decode_ug/wer_* | utils/best_wer.sh 
#%WER 7.99 [ 1002 / 12533, 74 ins, 153 del, 775 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_ug/wer_6

# and with per-utterance decoding:
# %WER 1.72 [ 216 / 12533, 26 ins, 45 del, 145 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_utt/wer_3
# %WER 8.40 [ 1053 / 12533, 85 ins, 158 del, 810 sub ] exp/nnet2_online_wsj/nnet_gpu_online/decode_ug_utt/wer_6

# And this is a suitable baseline: a system trained on RM only.
#a11:s5: grep WER exp/nnet2_online/nnet_gpu_online/decode/wer_* | utils/best_wer.sh 
#%WER 2.20 [ 276 / 12533, 25 ins, 69 del, 182 sub ] exp/nnet2_online/nnet_gpu_online/decode/wer_8
#a11:s5: grep WER exp/nnet2_online/nnet_gpu_online/decode_ug/wer_* | utils/best_wer.sh 
#%WER 10.14 [ 1271 / 12533, 127 ins, 198 del, 946 sub ] exp/nnet2_online/nnet_gpu_online/decode_ug/wer_11
