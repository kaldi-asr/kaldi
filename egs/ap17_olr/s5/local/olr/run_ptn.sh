#!/bin/bash

# this is a basic lstm script
# LSTM script runs for more epochs than the TDNN script
# and each epoch takes twice the time

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=0
affix=

# LSTM options
splice_indexes="-2,-1,0,1,2"  # default -2,-1,0,1,2 0 0
lstm_delay=" -1 "  # default -1 -2 -3
label_delay=5
num_lstm_layers=1  # default 3
cell_dim=1024 #512  # default 1024
hidden_dim=1024 #512  # default 1024
recurrent_projection_dim=256 #128  # default 256 
non_recurrent_projection_dim=256 #128 # default 256
chunk_width=20
chunk_left_context=40
chunk_right_context=0


# training options
num_epochs=10
initial_effective_lrate=0.0006
final_effective_lrate=0.00006
num_jobs_initial=2
num_jobs_final=12
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=false

#decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

#End configuration section

echo "$0 $@" # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

common_egs_dir=
dir=exp/nnet3/ptn

## PTN needs a neural acoustic model with fixed parameters, which can be trained with Thchs30 corpus
## we already provide a 6-layer raw TDNN using Thchs30
## and the nnet3 config file for initializing an LSTM LID model
mkdir -p $dir
cp -rf data/ptn_stuff/configs $dir
nnet3-init data/ptn_stuff/th30_tdnn_fix.raw exp/nnet3/ptn/configs/ptn.config exp/nnet3/ptn/0.raw

if [ $stage -le 8 ]; then
  local/olr/nnet3/train_lstm_raw_init.sh --stage $train_stage \
    --label-delay $label_delay \
    --lstm-delay "$lstm_delay" \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --cmd "$cpu_cmd" \
    --gpu-cmd "$gpu_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --chunk-right-context $chunk_right_context \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/train exp/olr_ali $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
    echo "---- evaluation PTN LID ----"
    for dev in dev_1s dev_3s dev_all; do
        local/olr/eval/lid_score.sh exp/nnet3/ptn data/$dev
    done
fi

exit 0;

