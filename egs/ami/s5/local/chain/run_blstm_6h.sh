#!/bin/bash

#adapted from swbd's local/chain/run_blstm_6h.sh script. We change the blstm dimensions

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
mic=ihm
use_ihm_ali=false
affix=
speed_perturb=true
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
chunk_width=150
chunk_left_context=40
chunk_right_context=40
xent_regularize=0.025

label_delay=0
# decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

remove_egs=false
common_egs_dir=

affix=
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


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments true || exit 1;


gmm=tri4a
if [ $use_ihm_ali == "true" ]; then
  gmm_dir=exp/ihm/$gmm
  mic=${mic}_cleanali
  ali_dir=${gmm_dir}_${mic}_train_parallel_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_parallel_sp_lats
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_${mic}_train_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_sp_lats
fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/$mic/chain/blstm${affix:+_$affix} # Note: _sp will get added to this if $speed_perturb == true.

if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix
train_set=train$suffix
latgen_train_set=$train_set

if [ $use_ihm_ali == "true" ]; then
  latgen_train_set=train_parallel$suffix
fi

treedir=exp/$mic/chain/tri5_2y_tree$suffix
lang=data/$mic/lang_chain_2y

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}


if [ $stage -le 10 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi


if [ $stage -le 11 ]; then
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

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 4200 data/$mic/$latgen_train_set $lang $ali_dir $treedir
fi

if [ $stage -le 13 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    --feat-dir data/$mic/${train_set}_hires \
    --ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --tree-dir $treedir \
    --splice-indexes="-2,-1,0,1,2 0 0" \
    --lstm-delay=" [-3,3] [-3,3] [-3,3] " \
    --xent-regularize $xent_regularize \
    --include-log-softmax false \
    --num-lstm-layers 3 \
    --cell-dim 512 \
    --hidden-dim 512 \
    --recurrent-projection-dim 128 \
    --non-recurrent-projection-dim 128 \
    --label-delay $label_delay \
    --self-repair-scale 0.00001 \
   $dir/configs || exit 1;

fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{8,9,10,11,12}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --chain.left-deriv-truncate 0 \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/$mic/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 16 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 250 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) &
  done
fi
wait;
exit 0;
