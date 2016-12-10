#!/bin/bash

set -o pipefail
set -e 
set -u

. cmd.sh

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
get_egs_stage=-10
egs_opts=

# LSTM options
splice_indexes="-2,-1,0,1,2 0"
label_delay=0
num_lstm_layers=2
cell_dim=64
hidden_dim=64
recurrent_projection_dim=32
non_recurrent_projection_dim=32
chunk_width=40
chunk_left_context=40
lstm_delay="-1 -2"

# training options
num_epochs=3
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
momentum=0.5
num_chunk_per_minibatch=256
samples_per_iter=20000
remove_egs=false
max_param_change=1

num_utts_subset_valid=6
num_utts_subset_train=6

use_dense_targets=false
extra_egs_copy_cmd="nnet3-copy-egs-overlap-detection ark:- ark:- |"

# target options
train_data_dir=data/sdm1/train_whole_sp_hires_bp
targets_scp=exp/sdm1/overlap_speech_train_cleaned_sp/overlap_feats.scp
deriv_weights_scp=exp/sdm1/overlap_speech_train_cleaned_sp/deriv_weights.scp
egs_dir=
nj=40
feat_type=raw
config_dir=
compute_objf_opts=

mic=sdm1
dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
if [ -z "$dir" ]; then
  dir=exp/$mic/nnet3_raw/nnet_lstm
fi

dir=$dir${affix:+_$affix}_n${num_hidden_layers}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi


if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
  
mkdir -p $dir

if [ $stage -le 3 ]; then
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay="$lstm_delay")
  steps/nnet3/lstm/make_raw_configs.py "${config_extra_opts[@]}" \
    --feat-dir=$train_data_dir --num-targets=3 \
    --splice-indexes="$splice_indexes" \
    --num-lstm-layers=$num_lstm_layers \
    --label-delay=$label_delay \
    --cell-dim=$cell_dim \
    --hidden-dim=$hidden_dim \
    --recurrent-projection-dim=$recurrent_projection_dim \
    --non-recurrent-projection-dim=$non_recurrent_projection_dim \
    --include-log-softmax=false --add-lda=false --add-idct=true \
    --add-final-sigmoid=true \
    --objective-type=xent \
    $dir/configs

fi

if [ $stage -le 4 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b{05,06,11,12}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp"
  fi

  egs_opts="$egs_opts --num-utts-subset-train $num_utts_subset_train --num-utts-subset-valid $num_utts_subset_valid"

  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --trainer.max-param-change=$max_param_change \
    ${config_dir:+--configs-dir=$config_dir} \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=1 \
    --use-gpu=true \
    --use-dense-targets=$use_dense_targets \
    --egs.extra-copy-cmd="$extra_egs_copy_cmd" \
    --feat-dir=$train_data_dir \
    --targets-scp="$targets_scp" \
    --dir=$dir || exit 1
fi

