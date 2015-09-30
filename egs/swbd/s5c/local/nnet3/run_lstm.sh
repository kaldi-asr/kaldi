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
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=20
clipping_threshold=5.0
norm_based_clipping=true
common_egs_dir=
has_fisher=true

# natural gradient options
ng_per_element_scale_options=
ng_affine_options=
num_epochs=10
# training options
initial_effective_lrate=0.0002
final_effective_lrate=0.00002
num_jobs_initial=1
num_jobs_final=12
shrink=0.99
momentum=0.9
num_chunk_per_minibatch=100
num_bptt_steps=20
samples_per_iter=20000
remove_egs=true
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

use_delay=false
if [ $label_delay -gt 0 ]; then use_delay=true; fi

dir=exp/nnet3/lstm${affix:+_$affix}${use_delay:+_ld$label_delay}
if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/tri4
else
  gmm_dir=exp/tri3
fi

ali_dir=${gmm_dir}_ali_nodup

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/lstm/train.sh --stage $train_stage \
    --label-delay $label_delay \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --shrink $shrink --momentum $momentum \
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
    data/train_nodup data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 8 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  num_jobs=`cat data/eval2000/utt2spk|cut -d' ' -f2|sort -u|wc -l`
  steps/nnet3/lstm/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
    --extra-left-context $chunk_left_context \
    $gmm_dir/graph_sw1_tg data/eval2000 \
    $dir/decode_eval2000_sw1_tg || exit 1;

  if $has_fisher; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg} || exit 1;
  fi

fi
wait;

exit 0;

