#!/bin/bash

#adapted from swbd's local/chain/6z.sh script. We change the TDNN config
# These are the other modifications:
#   1. AMI data has a lot of short segments so we use combine_short_segments.py
#      to combine contiguous segments to have a minimum_duration of 1.55 secs.
#      This is done to ensure that chain/get_egs.sh does not discard the
#      shorter segments
#   2. AMI data has a lot of alignment errors so we add an option to discard
#      segments with a lot of alignment errors. These are identified using
#      find_bad_utt.sh

# these are results with a slighly different min_seg_len=2secs and an older steps/cleanup/combine_short_segments.sh script
# also the trees for these results were built with min_seg_len=2 utterances, I moved tree building stage prior to segment
# combination, as I don't think it is affected by the minor change in alignments.
# Current experiments show that max-wer setting is not helpful when --use-ihm-ali=true
#---------------------------
##################### Systems with alignments from same acoustic data as features ###########################
  # Individual headset microphone systems
  #----------------------------------------
  # results might not change even with max-wer 45 (recommended)
  # local/chain/run_tdnn_ami_5.sh  --mic ihm --max-wer 50 --affix min_seg_len2_50wer (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 22.4 | 13098 94484 | 80.5 10.7 8.8 3.0 22.4 54.8 | 0.091 | exp/ihm/chain/tdnn_min_seg_len2_50wer_sp/decode_dev/ascore_10/dev_hires.ctm.filt.sys
  # %WER 22.4 | 12643 89973 | 80.3 12.6 7.1 2.8 22.4 53.2 | 0.155 | exp/ihm/chain/tdnn_min_seg_len2_50wer_sp/decode_eval/ascore_10/eval_hires.ctm.filt.sys

  # Multiple distant microphone systems
  #----------------------------------------
  # local/chain/run_tdnn_ami_5.sh  --mic mdm8 --affix msl1.5_45wer
  # %WER 38.5 | 14761 94496 | 65.5 17.6 16.9 4.0 38.5 66.5 | 0.620 | exp/mdm8/chain/tdnn_ami5_msl1.5_45wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 41.5 | 14219 89974 | 62.2 18.5 19.2 3.7 41.5 65.6 | 0.596 | exp/mdm8/chain/tdnn_ami5_msl1.5_45wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # Single distant microphone system
  #----------------------------------------
  # local/chain/run_tdnn_ami_5.sh  --mic sdm1 --affix msl1.5_45wer
  # %WER 42.8 | 14391 94487 | 60.8 19.3 19.9 3.6 42.8 69.1 | 0.588 | exp/sdm1/chain/tdnn_ami4_msl1.5_45wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 46.1 | 13754 89977 | 57.5 20.7 21.9 3.6 46.1 69.2 | 0.561 | exp/sdm1/chain/tdnn_ami4_msl1.5_45wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

##################### Systems with alignments from individual head-set microphone data ###########################
  # Multiple distant microphone system
  #----------------------------------------
  # local/chain/run_tdnn_ami_5.sh  --mic mdm8 --use-ihm-ali true --affix msl1.5_45wer
  # %WER 38.1 | 15296 94487 | 65.7 17.9 16.4 3.8 38.1 62.5 | 0.617 | exp/mdm8_cleanali/chain/tdnn_ami5_msl1.5_45wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 41.5 | 13795 89975 | 62.6 20.4 17.0 4.1 41.5 66.9 | 0.628 | exp/mdm8_cleanali/chain/tdnn_ami5_msl1.5_45wer_sp/decode_eval/ascore_8/eval_hires_o4.ctm.filt.sys

  # Single distant microphone system
  #----------------------------------------
  # results might not change even with max-wer 45 (recommended)
  # local/chain/run_tdnn_ami_5.sh  --mic sdm1 --use-ihm-ali true --max-wer 50 --affix msl1.5_50wer
  # %WER 41.6 | 14793 94504 | 61.8 19.3 18.9 3.4 41.6 65.3 | 0.591 | exp/sdm1_cleanali/chain/tdnn_ami4_msl1.5_50wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 45.4 | 14141 89972 | 57.9 20.7 21.4 3.3 45.4 64.8 | 0.567 | exp/sdm1_cleanali/chain/tdnn_ami4_msl1.5_50wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys


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
exp_name=tdnn_ami5


# training options
# chain options
frames_per_eg=150
xent_regularize=0.1
max_wer=

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
                                --frames-per-eg $frames_per_eg \
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

###################################

if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/tdnn/make_configs.py \
    --self-repair-scale-nonlinearity 0.00001 \
    --feat-dir $train_data_dir \
    --ivector-dir $train_ivector_dir \
    --tree-dir $treedir \
    --relu-dim 450 \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target 1.0 \
   $dir/configs || exit 1;
fi

if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
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
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
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
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --stage $decode_stage \
          --extra-left-context 20 \
          --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) &
  done
fi
wait;
exit 0;
