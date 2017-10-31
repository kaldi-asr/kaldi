#!/bin/bash

#    This is the standard "lstm" system, built in nnet3; this script
# is the version that's meant to run with data-cleanup, that doesn't
# support parallel alignments.


# by default:
# local/nnet3/run_lstm.sh

set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=7
min_seg_len=1.55
train_set=train
gmm=tri3b  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=  # cleanup affix for exp dirs, e.g. _cleaned

# Options which are not passed through to run_ivector_common.sh
affix=
common_egs_dir=
reporting_email=

# LSTM options
train_stage=-10
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
max_param_change=2.0

# training options
srand=0
num_epochs=6
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=15
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=true

#decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

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

if [ $stage -le 11 ]; then

    local/nnet3/run_ivector_common.sh --stage $stage \
                                      --nj $nj \
                                      --min-seg-len $min_seg_len \
                                      --train-set $train_set \
                                      --gmm $gmm \
                                      --num-threads-ubm $num_threads_ubm \
                                      --nnet3-affix "$nnet3_affix"
fi

gmm_dir=exp/${gmm}
graph_dir=$gmm_dir/graph_tg
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
dir=exp/nnet3${nnet3_affix}/lstm${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}_sp
train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
     $graph_dir/HCLG.fst $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs"
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay "$lstm_delay")
  steps/nnet3/lstm/make_configs.py  "${config_extra_opts[@]}" \
    --feat-dir $train_data_dir \
    --ivector-dir $train_ivector_dir \
    --ali-dir $ali_dir \
    --num-lstm-layers $num_lstm_layers \
    --splice-indexes "$splice_indexes " \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --label-delay $label_delay \
    --self-repair-scale-nonlinearity 0.00001 \
  $dir/configs || exit 1;
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/sprakbanken-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=1 \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  rm $dir/.error 2>/dev/null || true
   (
    steps/nnet3/decode.sh --nj 12 --cmd "$decode_cmd"  --num-threads 4 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_dev_hires \
      ${graph_dir} data/dev_hires ${dir}/decode_dev || exit 1
    steps/nnet3/decode.sh --nj 7 --cmd "$decode_cmd"  --num-threads 4 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_test_hires \
      ${graph_dir} data/test_hires ${dir}/decode_test || exit 1
    ) || touch $dir/.error &
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
