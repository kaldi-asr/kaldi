#!/bin/bash
#
# This script requires that you have run the toplevel run.sh script in TEDLIUM up to stage 7.
#
# Results: (Run for x in exp/chain/tdnn_sp/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null)
## %WER 13.9 | 507 17792 | 89.0 7.9 3.2 2.9 13.9 91.3 | 0.006 | exp/chain/tdnn_sp/decode_dev/score_8_1.0/ctm.filt.filt.sys
## %WER 13.0 | 507 17792 | 89.9 6.8 3.4 2.8 13.0 90.1 | 0.017 | exp/chain/tdnn_sp/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 13.8 | 1155 27512 | 88.9 7.3 3.9 2.7 13.8 86.9 | 0.099 | exp/chain/tdnn_sp/decode_test/score_10_0.5/ctm.filt.filt.sys
## %WER 12.9 | 1155 27512 | 90.1 6.5 3.4 3.0 12.9 86.8 | 0.006 | exp/chain/tdnn_sp/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys
# The final WER (rescored WER on the test set) is what we are interested in.

# To reproduce the setup used in the paper, set the following variables:
# dir=exp/chain/tdnn_more_ce 
# relu_dim=525
# xent_regularize=0.2
#
# Results: (Run for x in exp/chain/tdnn_more_ce_sp/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null)
## %WER 14.3 | 507 17792 | 89.0 7.8 3.2 3.3 14.3 93.5 | 0.116 | exp/chain/tdnn_more_ce_sp/decode_dev/score_10_0.0/ctm.filt.filt.sys
## %WER 13.0 | 507 17792 | 90.0 6.9 3.2 2.9 13.0 91.3 | -0.003 | exp/chain/tdnn_more_ce_sp/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 13.8 | 1155 27512 | 89.1 7.4 3.4 2.9 13.8 87.5 | 0.082 | exp/chain/tdnn_more_ce_sp/decode_test/score_10_0.5/ctm.filt.filt.sys
## %WER 12.8 | 1155 27512 | 90.4 6.6 3.1 3.1 12.8 86.7 | 0.014 | exp/chain/tdnn_more_ce_sp/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

set -uo pipefail

# configs for 'chain'
affix=
stage=0 # After running the entire script once, you can set stage=12 to tune the neural net only.
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
self_repair_scale=0.00001
# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=8
minibatch_size=128
relu_dim=425
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1


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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
train_set=train$suffix
gmm=tri3
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali$suffix
lats_dir=${ali_dir/ali/lats}
treedir=exp/chain/tri3_tree$suffix
lang=data/lang_chain

mkdir -p $dir
# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat ${ali_dir}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang $gmm_dir $lats_dir
  rm ${lats_dir}/fsts.*.gz # save space
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
      --cmd "$train_cmd" 4200 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";
  dim_opts="--relu-dim $relu_dim"

  # create the config files for nnet initialization
  repair_opts=${self_repair_scale:+" --self-repair-scale $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py \
    $repair_opts \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target $final_layer_normalize_target \
    $dir/configs || exit 1;
fi

if [ $stage -le 13 ]; then
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  touch $dir/egs/.nodelete

 steps/nnet3/chain/train.py --stage $train_stage \
   --cmd "$decode_cmd" \
   --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
   --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
   --chain.xent-regularize $xent_regularize \
   --chain.leaky-hmm-coefficient 0.1 \
   --chain.l2-regularize 0.00005 \
   --chain.apply-deriv-weights false \
   --chain.lm-opts="--num-extra-lm-states=2000" \
   --egs.stage $get_egs_stage \
   --egs.opts "--frames-overlap-per-eg 0" \
   --egs.chunk-width $frames_per_eg \
   --trainer.num-chunk-per-minibatch $minibatch_size \
   --trainer.frames-per-iter 1500000 \
   --trainer.num-epochs $num_epochs \
   --trainer.optimization.num-jobs-initial $num_jobs_initial \
   --trainer.optimization.num-jobs-final $num_jobs_final \
   --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
   --trainer.optimization.final-effective-lrate $final_effective_lrate \
   --trainer.max-param-change $max_param_change \
   --cleanup.remove-egs $remove_egs \
   --feat-dir data/${train_set}_hires \
   --tree-dir $treedir \
   --lat-dir exp/tri3_lats$suffix \
   --dir $dir || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 14 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  for decode_set in dev test; do
    (
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 8 --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      --scoring-opts "--min-lmwt 8 --max-lmwt 12" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1;

    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test data/lang_rescore data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter} \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_rescore || exit 1;
    ) &
  done
fi

wait