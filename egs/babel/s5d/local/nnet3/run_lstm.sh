#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#           2015  Vijayaditya Peddinti
#           2015  Xingyu Na
#           2015  Pegah Ghahrmani
# Apache 2.0.


# this is a basic lstm script
# LSTM script runs for more epochs than the TDNN script
# and each epoch takes twice the time

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=-10
has_fisher=true
affix=
speed_perturb=true
common_egs_dir=

# LSTM options
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
lstm_delay=" -1 -2 -3"
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=40
chunk_right_context=0

# training options
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2
num_jobs_final=6
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

#decode options
extra_left_context=
frames_per_chunk=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh
. ./cmd.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/lstm
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=train$suffix
ali_dir=exp/tri5_ali$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
	--speed-perturb $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/lstm/train.sh --stage $train_stage \
    --label-delay $label_delay \
    --lstm-delay "$lstm_delay" \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --cmd "$decode_cmd" \
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
    data/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
  touch $dir/.done
fi

wait;
exit 0;
