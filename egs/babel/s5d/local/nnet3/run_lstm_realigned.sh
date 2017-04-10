#!/bin/bash

stage=0
train_stage=-10
affix=
speed_perturb=true
multicondition=false
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
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2
num_jobs_final=6
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

align_model_dir=exp/nnet3/tdnn_sp
extra_align_opts=
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

dir=exp/nnet3/lstm_realigned
if [ "$multicondition" == "true" ]; then
  suffix=${suffix}_mc
  dir=exp/nnet3${multicondition:+_multicondition}/lstm
fi


suffix=${suffix}
dir=$dir${affix:+_$affix}$suffix
train_set=train$suffix


if [ "$multicondition" == "true" ]; then
  ivector_dir=exp/nnet3_multicondition/ivectors_${train_set}
else
  ivector_dir=exp/nnet3/ivectors_${train_set}
fi

# think of  a better way to determine ali_dir name
ali_dir=${align_model_dir}_${train_set}_ali
if [ $stage -le 1 ]; then
  steps/nnet3/align.sh  --cmd "$decode_cmd" --use-gpu false \
    $extra_align_opts --online-ivector-dir $ivector_dir \
    --nj 400 data/${train_set}_hires data/lang \
    $align_model_dir $ali_dir || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs";
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay "$lstm_delay")
  steps/nnet3/lstm/make_configs.py  "${config_extra_opts[@]}" \
    --feat-dir data/${train_set}_hires \
    --ivector-dir $ivector_dir \
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

if [ $stage -le 3 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/babel-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.cv-minibatch-size 128 \
    --trainer.optimization.momentum=$momentum \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=20 \
    --use-gpu=true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

wait;
exit 0;
