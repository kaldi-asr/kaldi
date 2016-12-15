#!/bin/bash


# this is a basic lstm script
# it is also used to run blstm experiments, so it uses a lot more variables
# than a typical top level script.

# based on egs/fisher_swbd/s5/local/nnet3/run_lstm.sh

stage=7
train_stage=-10
egs_stage=
affix=
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
chunk_width=20
chunk_left_context=40
chunk_right_context=0


# training options
num_epochs=6
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=4
num_jobs_final=22
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

#decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  cmd_opts=" --config conf/queue_only_k80.conf --only-k80 false"
fi


dir=exp/nnet3/lstm
dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
ali_dir=exp/tri5a_rvb_ali

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay "$lstm_delay")
  steps/nnet3/lstm/make_configs.py  "${config_extra_opts[@]}" \
    --feat-dir data/train_rvb_hires \
    --ivector-dir exp/nnet3/ivectors_train \
    --ali-dir $ali_dir \
    --num-lstm-layers $num_lstm_layers \
    --splice-indexes "$splice_indexes " \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --label-delay $label_delay \
    --self-repair-scale-nonlinearity 0.00001 \
    --self-repair-scale-clipgradient 1.0 \
   $dir/configs || exit 1;

fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd $cmd_opts" \
    --feat.online-ivector-dir=exp/nnet3/ivectors_train \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --egs.opts " --nj 12 " \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --egs.stage "$egs_stage" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=100 \
    --use-gpu=true \
    --feat-dir=data/train_rvb_hires \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

wait;
exit 0;
