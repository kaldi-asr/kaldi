#!/bin/bash

# this is a basic lstm script

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
has_fisher=true
mic=ihm
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=

# lstm options
splice_indexes="-2,-1,0,1,2"
num_lstm_layers=1
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=20
norm_based_clipping=false
clipping_threshold=15

# natural gradient options
ng_per_element_scale_options=
ng_affine_options=

# training options
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_chunk_per_minibatch=100
samples_per_iter=20000

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

dir=exp/$mic/nnet3/lstm${speed_perturb:+_sp}${affix:+_$affix}
if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/$mic/tri4a
else
  gmm_dir=exp/$mic/tri3a
fi

if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
  ali_dir=${gmm_dir}_sp_ali
else
  train_set=train
  ali_dir=${gmm_dir}_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}

local/nnet3/run_ivector_common.sh --stage $stage \
  --use-sat-alignments $use_sat_alignments \
  --speed-perturb $speed_perturb || exit 1;

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/lstm/train.sh --stage $train_stage \
    --num-epochs 3 --num-jobs-initial 2 --num-jobs-final 12 \
    --egs-dir "$common_egs_dir" \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --cmd "$decode_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --norm-based-clipping $norm_based_clipping \
    --clipping-threshold $clipping_threshold \
    --ng-per-element-scale-options "$ng_per_element_scale_options" \
    --ng-affine-options "$ng_affine_options" \
    data/$mic/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
fi
exit;
if [ $stage -le 8 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/$mic/nnet3/extractor "$dir" ${dir}_online || exit 1;
fi
wait;

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt
      steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector (--online false)
  for decode_set in dev eval; do
    (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt_offline
      steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true --online false $graph_dir data/$mic/${decode_set}_hires \
          $decode_dir || exit 1;
    ) & 
  done
fi
wait;

exit 0;

