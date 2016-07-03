#!/bin/bash

#This is a state preserving lstm script

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=9
train_stage=-10
has_fisher=true
affix=state_preserving
speed_perturb=true
common_egs_dir=
reporting_email=

# LSTM options
splice_indexes="-2,-1,0,1,2 0 0"
lstm_delay=" -1 -2 -3 "
label_delay=5
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=200
chunk_left_context=40
chunk_right_context=0
minibatch_chunk_size=20
left_shift_window=true

# training options
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=15
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=false


# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $minibatch_chunk_size -gt 0 ]; then
  samples_per_iter=$[$samples_per_iter * $minibatch_chunk_size / $chunk_width]
fi

local/nnet3/run_lstm.sh --affix $affix \
  --has-fisher $has_fisher \
  --speed-perturb $speed_perturb \
  --stage $stage \
  --train-stage $train_stage \
  --splice-indexes "$splice_indexes " \
  --label-delay $label_delay \
  --lstm-delay "$lstm_delay" \
  --num-lstm-layers $num_lstm_layers \
  --num-jobs-initial $num_jobs_initial \
  --num-jobs-final $num_jobs_final \
  --num-epochs $num_epochs \
  --initial-effective-lrate $initial_effective_lrate \
  --final-effective-lrate $final_effective_lrate \
  --cell-dim $cell_dim \
  --hidden-dim $hidden_dim \
  --recurrent-projection-dim $recurrent_projection_dim \
  --non-recurrent-projection-dim $non_recurrent_projection_dim \
  --momentum $momentum \
  --num-chunk-per-minibatch $num_chunk_per_minibatch \
  --chunk-width $chunk_width \
  --chunk-left-context $chunk_left_context \
  --chunk-right-context $chunk_right_context \
  --common-egs-dir "$common_egs_dir" \
  --remove-egs $remove_egs \
  --reporting-email "$reporting_email" \
  --minibatch-chunk-size $minibatch_chunk_size \
  --samples-per-iter $samples_per_iter \
  --left-shift-window $left_shift_window \

exit 0;
