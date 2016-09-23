#!/bin/bash
#
# This script requires that you have run the toplevel run.sh script in TEDLIUM up to stage 7.
#
# Results: (Run for x in exp/chain/tdnn/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null)
## Number of parameters: 6172530
## %WER 14.1 | 507 17792 | 88.6 7.3 4.1 2.7 14.1 92.9 | 0.075 | exp/chain/tdnn/decode_dev/score_10_0.5/ctm.filt.filt.sys
## %WER 13.3 | 507 17792 | 89.7 6.9 3.4 2.9 13.3 92.1 | 0.000 | exp/chain/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 13.8 | 1155 27512 | 89.4 7.5 3.1 3.2 13.8 87.9 | 0.101 | exp/chain/tdnn/decode_test/score_10_0.0/ctm.filt.filt.sys
## %WER 12.9 | 1155 27512 | 90.1 6.6 3.3 2.9 12.9 86.1 | 0.043 | exp/chain/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys
# The final WER (rescored WER on the test set) is what we are interested in.

# To reproduce the setup used in the paper, set the following variables:
# affix=_more_ce
# relu_dim=525
# xent_regularize=0.2
#
# Results: (Run for x in exp/chain/tdnn_more_ce/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null)
## Number of parameters: 8758742
## %WER 14.3 | 507 17792 | 89.0 7.8 3.2 3.3 14.3 93.5 | 0.116 | exp/chain/tdnn_more_ce/decode_dev/score_10_0.0/ctm.filt.filt.sys
## %WER 13.0 | 507 17792 | 90.0 6.9 3.2 2.9 13.0 91.3 | -0.003 | exp/chain/tdnn_more_ce/decode_devv_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 13.8 | 1155 27512 | 89.1 7.4 3.4 2.9 13.8 87.5 | 0.082 | exp/chain/tdnn_more_ce/decode_test/score_10_0.5/ctm.filt.filt.sys
## %WER 12.8 | 1155 27512 | 90.4 6.6 3.1 3.1 12.8 86.7 | 0.014 | exp/chain/tdnn_more_ce/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

set -uo pipefail

# configs for 'chain'
affix=
stage=0 # After running the entire script once, you can set stage=12 to tune the neural net only.
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn
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
xent_regularize=0.1


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=${dir}${affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 9" if you have already
# run those things.

gmm_dir=exp/tri3
ali_dir=exp/tri3_ali_sp
lats_dir=${ali_dir/ali/lats} # note, this is a search-and-replace from 'ali' to 'lats'
treedir=exp/chain/tri3_tree
lang=data/lang_chain

mkdir -p $dir

local/nnet3/run_ivector_common.sh --stage $stage \
  --generate-alignments false \
  --speed-perturb true || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat ${ali_dir}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_sp \
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
      --cmd "$train_cmd" 4000 data/train_sp $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py \
    $repair_opts \
    --feat-dir data/train_sp_hires \
    --ivector-dir exp/nnet3/ivectors_train_sp \
    --tree-dir $treedir \
    --relu-dim $relu_dim \
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
   --feat.online-ivector-dir exp/nnet3/ivectors_train_sp \
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
   --cleanup.preserve-model-interval 20 \
   --feat-dir data/train_sp_hires \
   --tree-dir $treedir \
   --lat-dir $lats_dir \
   --dir $dir || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  for decode_set in dev test; do
    (
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $(wc -l < data/$decode_set/spk2utt) --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      --scoring-opts "--min_lmwt 5 --max_lmwt 15" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1;

    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test data/lang_rescore data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter} \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_rescore || exit 1;
    ) &
  done
fi

wait
