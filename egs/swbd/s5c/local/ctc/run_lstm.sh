#!/bin/bash

stage=0
train_stage=-10
affix=
speed_perturb=true
common_egs_dir=

# LSTM options
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=50
chunk_left_context=40
clipping_threshold=30.0
norm_based_clipping=true

# natural gradient options
ng_per_element_scale_options=
ng_affine_options=
num_epochs=2

# training options
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2
num_jobs_final=12
momentum=0.5
adaptive_shrink=true
shrink=0.98
num_chunk_per_minibatch=100
num_bptt_steps=
frames_per_iter=400000
remove_egs=true

#decode options
extra_left_context=
frames_per_chunk=

# ctc options
treedir=exp/ctc/tri5b_tree

# End configuration section.

echo "$0 $@"  # Print the command line for logging

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

use_delay=false
if [ $label_delay -gt 0 ]; then use_delay=true; fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/$mic/ctc/lstm
dir=$dir${affix:+_$affix}${use_delay:+_ld$label_delay}
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments false || exit 1;

if [ $stage -le 9 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file.
  lang=data/lang_ctc
  rm -r $lang 2>/dev/null
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  utils/gen_topo.pl 1 1 $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 10 ]; then
  # Starting from the alignments in tri4_ali_nodup*, we train a rudimentary
  # LDA+MLLT system with a 1-state HMM topology and with only left phonetic
  # context (one phone's worth of left context, for now).  We set "--num-iters
  # 1" because we only need the tree from this system.
  steps/train_sat.sh --cmd "$train_cmd" --num-iters 1 \
    --tree-stats-opts "--collapse-pdf-classes=true" \
    --cluster-phones-opts "--pdf-class-list=0" \
    --context-opts "--context-width=2 --central-position=1" \
     2500 15000 data/$train_set data/lang_ctc $ali_dir $treedir
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/ctc/train_lstm.sh --stage $train_stage \
    --label-delay $label_delay \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --frames-per-iter $frames_per_iter \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --adaptive-shrink "$adaptive_shrink" \
    --shrink $shrink \
    --cmd "$decode_cmd" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --clipping-threshold $clipping_threshold \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --num-bptt-steps "$num_bptt_steps" \
    --norm-based-clipping $norm_based_clipping \
    --ng-per-element-scale-options "$ng_per_element_scale_options" \
    --ng-affine-options "$ng_affine_options" \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/${train_set}_hires data/lang_ctc $treedir exp/tri4_lats_nodup$suffix $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  phone_lm_weight=0.15
  steps/nnet3/ctc/mkgraph.sh --phone-lm-weight $phone_lm_weight \
      data/lang_lang_sw1_tg $dir $dir/graph_sw1_tg_${phone_lm_weight}
fi

decode_suff=sw1_tg_${phone_lm_weight}
graph_dir=exp/tri4/graph_${decode_suff}
if [ $stage -le 14 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  for decode_set in train_dev_hires eval2000_hires; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/ctc/decode.sh --nj 250 --cmd "$decode_cmd" \
          --extra-left-context $extra_left_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
