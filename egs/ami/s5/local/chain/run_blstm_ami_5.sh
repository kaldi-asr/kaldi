#!/bin/bash


###
# Does not give improvements over xent+blstm system !!
#local/chain/run_blstm_ami_5.sh --mic sdm1 --use-ihm-ali false --max-wer 45 --affix msl1.5_45wer
# %WER 42.5 | 14769 94491 | 61.0 19.9 19.1 3.5 42.5 67.5 | 0.605 | exp/sdm1/chain/blstm_ami5_msl1.5_45wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
# %WER 45.7 | 13674 89971 | 57.7 21.0 21.3 3.5 45.7 69.1 | 0.572 | exp/sdm1/chain/blstm_ami5_msl1.5_45wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

set -e

# configs for 'chain'
stage=10
train_stage=-10
get_egs_stage=-10
decode_stage=1
mic=ihm
use_ihm_ali=false
affix=
common_egs_dir=
exp_name=blstm_ami5

# LSTM options
chunk_width=150
chunk_left_context=40
chunk_right_context=40


# decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

# training options
# chain options
xent_regularize=0.1
max_wer=45

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


new_mic=$mic
if [ $use_ihm_ali == "true" ]; then
  new_mic=${mic}_cleanali
fi

# we do speed perturbation by default, however for sake of
# consistency with older naming convention we  add _sp

dir=exp/$new_mic/chain/${exp_name}${affix:+_$affix}_sp
treedir=exp/$new_mic/chain/tri5_2y_tree_sp
lang=data/$new_mic/lang_chain_2y

local/chain/run_chain_common.sh --stage $stage \
                                --mic $mic \
                                --use-ihm-ali $use_ihm_ali \
                                --frames-per-eg $chunk_width \
                                --max-wer $max_wer \
                                --dir $dir \
                                --treedir $treedir \
                                --lang $lang
mic=$new_mic
. $dir/vars
# sets the directory names where features, ivectors and lattices are stored
#train_data_dir
#train_ivector_dir
#lat_dir

if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    --feat-dir $train_data_dir \
    --ivector-dir $train_ivector_dir \
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
    --label-delay 0 \
    --self-repair-scale 0.00001 \
   $dir/configs || exit 1;

fi

if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{8,9,10,11,12}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 5 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}
if [ $stage -le 18 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 19 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --stage $decode_stage \
          --nj $num_jobs --cmd "$decode_cmd" \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) &
  done
fi
wait;
exit 0;


