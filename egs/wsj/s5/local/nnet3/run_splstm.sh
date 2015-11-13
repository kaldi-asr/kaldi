#!/bin/bash

# this is a state preserving lstm script

# Note: there is a script steps/cleanup/combine_short_segments.sh, which combines
# utterances shorter than a specified length (e.g 2sec) to have the minimum length.
# It helps us not to lose data that is too short (utterances of too short length
# will be discarded). But we do not include it in the current recipe because
# we have not been able to observe the performance gains by including it.

stage=0
train_stage=-10
affix=state_preserving
common_egs_dir=

# LSTM options
# In state preserving mode chunk_width is the size of the original (larger) chunk,
# which is to be split into several smaller chunks of size minibatch_chunk_size 
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


# training options
initial_effective_lrate=0.0006
final_effective_lrate=0.00006
num_jobs_initial=2
num_jobs_final=12
samples_per_iter=2000
remove_egs=true
minibatch_chunk_size=20
left_shift_window=true

#End configuration section

echo "$0 $@" # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


local/nnet3/run_lstm.sh --affix $affix \
  --stage $stage \
  --train-stage $train_stage \
  --label-delay $label_delay \
  --lstm-delay "$lstm_delay" \
  --num-lstm-layers $num_lstm_layers \
  --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
  --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
  --cell-dim $cell_dim \
  --hidden-dim $hidden_dim \
  --recurrent-projection-dim $recurrent_projection_dim \
  --non-recurrent-projection-dim $non_recurrent_projection_dim \
  --chunk-width $chunk_width \
  --chunk-left-context $chunk_left_context \
  --chunk-right-context $chunk_right_context \
  --common-egs-dir "$common_egs_dir" \
  --remove-egs $remove_egs \
  --minibatch-chunk-size $minibatch_chunk_size \
  --samples-per-iter $samples_per_iter \
  --left-shift-window $left_shift_window

