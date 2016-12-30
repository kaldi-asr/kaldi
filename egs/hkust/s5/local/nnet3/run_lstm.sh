#!/bin/bash

# this is a basic lstm script

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
use_sat_alignments=true
affix=
speed_perturb=true

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
clipping_threshold=10.0
norm_based_clipping=true
common_egs_dir=

# natural gradient options
ng_per_element_scale_options=
ng_affine_options=
num_epochs=4

# training options
initial_effective_lrate=0.0002
final_effective_lrate=0.00002
num_jobs_initial=2
num_jobs_final=12
shrink=0.98
momentum=0.5
adaptive_shrink=true
num_chunk_per_minibatch=100
num_bptt_steps=20
samples_per_iter=20000
remove_egs=true

# feature options
use_ivectors=true

#decode options
extra_left_context=
frames_per_chunk=

# End configuration section.

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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=exp/nnet3/lstm
dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix

if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/tri5a
else
  gmm_dir=exp/tri3a
fi
train_set=train$suffix
ali_dir=${gmm_dir}${suffix}_ali
graph_dir=$gmm_dir/graph

if [ $stage -le 7 ]; then
  local/nnet3/run_ivector_common.sh --stage $stage \
    --use-sat-alignments $use_sat_alignments \
    --speed-perturb $speed_perturb || exit 1;
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/hkust-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  if [ "$use_ivectors" == "true" ]; then
    ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${train_set}_hires "
    cmvn_opts="--norm-means=false --norm-vars=false"
  else
    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=true"
  fi

  steps/nnet3/lstm/train.sh $ivector_opts --stage $train_stage \
    --label-delay $label_delay \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "$cmvn_opts" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --shrink $shrink --momentum $momentum \
    --adaptive-shrink "$adaptive_shrink" \
    --lstm-delay "$lstm_delay" \
    --cmd "$decode_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --clipping-threshold $clipping_threshold \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --num-bptt-steps $num_bptt_steps \
    --norm-based-clipping $norm_based_clipping \
    --ng-per-element-scale-options "$ng_per_element_scale_options" \
    --ng-affine-options "$ng_affine_options" \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev; do
      (
      num_jobs=`cat data/${decode_set}/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}
      if [ "$use_ivectors" == "true" ]; then
        ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${decode_set} "
      else
        ivector_opts=
      fi

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" $ivector_opts \
        --extra-left-context $extra_left_context \
        --frames-per-chunk "$frames_per_chunk" \
        $graph_dir data/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;

exit 0;

