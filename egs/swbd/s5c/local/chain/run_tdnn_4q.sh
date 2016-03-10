#!/bin/bash

# this is based on Dan's tdnn_2o script
# it has a different splicing configuration
# it uses the PerDimensionWeightedAverage pooling in place of the Jesus layer

set -e

#%WER 11.1 | 1831 21395 | 90.2 6.3 3.5 1.3 11.1 46.6 | exp/chain/tdnn_v1_trial6_sp/decode_eval2000_sw1_fsh_fg/score_12_0.0/eval2000_hires.ctm.swbd.filt.sys
#%WER 16.6 | 4459 42989 | 85.2 9.5 5.3 1.8 16.6 53.4 | exp/chain/tdnn_v1_trial6_sp/decode_eval2000_sw1_fsh_fg/score_11_0.0/eval2000_hires.ctm.filt.sys
#%WER 15.59 [ 7671 / 49204, 883 ins, 2234 del, 4554 sub ] exp/chain/tdnn_v1_trial6_sp/decode_train_dev_sw1_fsh_fg/wer_12_0.0


# configs for 'chain'
affix=
stage=10
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_4q  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# TDNN options
splice_indexes="-2,-1,0,1,2 -1,0,2 -3,0,3 -6,-3,0,3 -6,-3,0,3"
# smoothing options
pool_window=7 
pool_type='per-dim-weighted-average'
pool_lpfilter_width=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=1.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=64
relu_dim=700
frames_per_eg=150
remove_egs=false
common_egs_dir=



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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_2o_tree$suffix
lang=data/lang_chain_2o


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train_tdnn.sh --stage $train_stage \
    --apply-deriv-weights false \
    --lm-opts "--num-extra-lm-states=2000" \
    --get-egs-stage $get_egs_stage \
    --minibatch-size $minibatch_size \
    --egs-opts "--frames-overlap-per-eg 0" \
    --frames-per-eg $frames_per_eg \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --splice-indexes "$splice_indexes" \
    --pool-type "$pool_type" \
    --pool-window "$pool_window" \
    --pool-lpfilter-width "$pool_lpfilter_width" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --final-layer-normalize-target $final_layer_normalize_target \
    --relu-dim $relu_dim \
    --cmd "$decode_cmd" \
    --remove-egs $remove_egs \
    --egs-dir "$common_egs_dir" \
    data/${train_set}_hires $treedir exp/tri4_lats_nodup$suffix $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 14 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
