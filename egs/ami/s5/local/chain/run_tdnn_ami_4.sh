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
  # local/chain/run_tdnn_ami_4.sh  --mic ihm --affix min_seg_len2 (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 22.6 | 13098 94486 | 80.3 10.6 9.1 2.9 22.6 54.9 | 0.086 | exp/ihm/chain/tdnn_min_seg_len2_sp/decode_dev/ascore_10/dev_hires.ctm.filt.sys
  # %WER 22.5 | 12643 89975 | 80.2 12.3 7.4 2.7 22.5 52.8 | 0.153 | exp/ihm/chain/tdnn_min_seg_len2_sp/decode_eval/ascore_10/eval_hires.ctm.filt.sys

  # local/chain/run_tdnn_ami_4.sh  --mic ihm --max-wer 50 --affix min_seg_len2_50wer (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 22.4 | 13098 94484 | 80.5 10.7 8.8 3.0 22.4 54.8 | 0.091 | exp/ihm/chain/tdnn_min_seg_len2_50wer_sp/decode_dev/ascore_10/dev_hires.ctm.filt.sys
  # %WER 22.4 | 12643 89973 | 80.3 12.6 7.1 2.8 22.4 53.2 | 0.155 | exp/ihm/chain/tdnn_min_seg_len2_50wer_sp/decode_eval/ascore_10/eval_hires.ctm.filt.sys

  # Multiple distant microphone systems
  #----------------------------------------
  # local/chain/run_tdnn_ami_4.sh  --mic mdm8 --affix min_seg_len2 (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 40.0 | 15332 94473 | 63.1 15.8 21.0 3.2 40.0 63.9 | 0.604 | exp/mdm8/chain/tdnn_min_seg_len2_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 43.2 | 13630 89954 | 59.8 16.5 23.8 3.0 43.2 68.0 | 0.587 | exp/mdm8/chain/tdnn_min_seg_len2_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # local/chain/run_tdnn_ami_4.sh  --mic mdm8 --max-wer 50 --affix min_seg_len2_50wer (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 38.8 | 14180 94493 | 65.1 17.2 17.7 3.8 38.8 69.4 | 0.617 | exp/mdm8/chain/tdnn_min_seg_len2_50wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 42.1 | 13960 89974 | 61.4 18.0 20.6 3.4 42.1 66.2 | 0.591 | exp/mdm8/chain/tdnn_min_seg_len2_50wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # Single distant microphone systems
  #----------------------------------------
  # local/chain/run_tdnn_ami_4.sh  --mic sdm1 --affix min_seg_len2 (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 43.5 | 14629 94479 | 59.5 17.6 22.9 3.0 43.5 67.8 | 0.577 | exp/sdm1/chain/tdnn_min_seg_len2_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 47.3 | 13210 89956 | 55.9 19.1 25.1 3.2 47.3 71.8 | 0.555 | exp/sdm1/chain/tdnn_min_seg_len2_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # local/chain/run_tdnn_ami_4.sh  --mic sdm1 --max-wer 50 --affix min_seg_len2_50wer (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 42.8 | 14535 94491 | 60.7 18.7 20.6 3.6 42.8 68.4 | 0.583 | exp/sdm1/chain/tdnn_min_seg_len2_50wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 46.4 | 13101 89971 | 57.0 20.3 22.6 3.4 46.4 72.1 | 0.555 | exp/sdm1/chain/tdnn_min_seg_len2_50wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

##################### Systems with alignments from individual head-set microphone data ###########################
  # Multiple distant microphone systems
  #----------------------------------------
  # local/chain/run_tdnn_ami_4.sh  --mic mdm8 --use-ihm-ali true --affix min_seg_len2 (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 38.2 | 14638 94499 | 65.3 17.1 17.6 3.5 38.2 65.7 | 0.616 | exp/mdm8_cleanali/chain/tdnn_min_seg_len2_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 42.0 | 14054 89957 | 61.0 17.6 21.4 3.0 42.0 64.8 | 0.596 | exp/mdm8_cleanali/chain/tdnn_min_seg_len2_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # local/chain/run_tdnn_ami_4.sh  --mic mdm8 --use-ihm-ali true --max-wer 50
  # running

  # Single distant microphone systems
  #----------------------------------------
  # local/chain/run_tdnn_ami_4.sh  --mic sdm1 --use-ihm-ali true --affix min_seg_len2 (built with min-seg-len 2 secs, but script now just supports (frames_per_eg+5)/100)
  # %WER 41.3 | 14041 94506 | 62.1 19.4 18.5 3.4 41.3 69.2 | 0.587 | exp/sdm1_cleanali/chain/tdnn_min_seg_len2_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 45.3 | 14052 89969 | 57.9 20.9 21.2 3.2 45.3 66.2 | 0.563 | exp/sdm1_cleanali/chain/tdnn_min_seg_len2_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys

  # local/chain/run_tdnn_ami_4.sh  --mic sdm1 --use-ihm-ali true --max-wer 50 --affix msl1.5_50wer (built with min-seg-len (frames_per_eg+5)/100)
  # %WER 41.6 | 14793 94504 | 61.8 19.3 18.9 3.4 41.6 65.3 | 0.591 | exp/sdm1_cleanali/chain/tdnn_ami4_msl1.5_50wer_sp/decode_dev/ascore_9/dev_hires_o4.ctm.filt.sys
  # %WER 45.4 | 14141 89972 | 57.9 20.7 21.4 3.3 45.4 64.8 | 0.567 | exp/sdm1_cleanali/chain/tdnn_ami4_msl1.5_50wer_sp/decode_eval/ascore_9/eval_hires_o4.ctm.filt.sys


set -e

# configs for 'chain'
stage=10
train_stage=-10
get_egs_stage=-10
mic=ihm
use_ihm_ali=false
affix=
common_egs_dir=
exp_name=tdnn_ami4


# training options
# chain options
frames_per_eg=150
xent_regularize=0.1
max_wer=

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

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments true || exit 1;


# Set the variables. These are based on variables set by run_ivector_common.sh
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

dir=exp/$mic/chain/${exp_name}${affix:+_$affix} # Note: _sp will get added to this if $speed_perturb == true.
dir=${dir}_sp # we do speed perturbation by default, however for sake of consistency with older naming convention
              # we  add _sp


treedir=exp/$mic/chain/tri5_2y_tree_sp
lang=data/$mic/lang_chain_2y

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}

train_set=train_sp
latgen_train_set=train_sp
if [ $use_ihm_ali == "true" ]; then
  latgen_train_set=train_parallel_sp
fi

###################################

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
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4200 data/$mic/$latgen_train_set $lang $ali_dir $treedir

  # this is done to prevent accidental overwrites of the tree
  chmod -R 555 $treedir
fi



# combining the segments in training data to have a minimum length of frames_per_eg + tolerance
# this is critical stage in AMI (gives 1% absolute improvement)
min_seg_len=$(python -c "print ($frames_per_eg+5)/100.0")
if [ $stage -le 12 ]; then
  rm -rf data/$mic/${train_set}_min${min_seg_len}_hires
  steps/cleanup/combine_short_segments.py --minimum-duration $min_seg_len \
    --input-data-dir data/$mic/${train_set}_hires \
    --output-data-dir data/$mic/${train_set}_min${min_seg_len}_hires

  #extract ivectors for the new data
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/$mic/${train_set}_min${min_seg_len}_hires data/$mic/${train_set}_min${min_seg_len}_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/$mic/${train_set}_min${min_seg_len}_hires_max2 \
    exp/$mic/nnet3/extractor \
    exp/$mic/nnet3/ivectors_${train_set}_min${min_seg_len} || exit 1;

 # combine the non-hires features for alignments/lattices
 rm -rf data/$mic/${latgen_train_set}_min${min_seg_len}
 steps/cleanup/combine_short_segments.py --minimum-duration $min_seg_len \
                   --input-data-dir data/$mic/${latgen_train_set} \
                   --output-data-dir data/$mic/${latgen_train_set}_min${min_seg_len}
fi

train_set=${train_set}_min${min_seg_len}
latgen_train_set=${latgen_train_set}_min${min_seg_len}
ivector_dir=exp/$mic/nnet3/ivectors_${train_set}
ali_dir=${ali_dir}_min${min_seg_len}
lat_dir=${lat_dir}_min${min_seg_len}
if [ $stage -le 13 ]; then
  # realigning data as the segments would have changed
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set data/lang $gmm_dir $ali_dir || exit 1;
fi

if [ $stage -le 14 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

mkdir -p $dir
train_data_dir=data/$mic/${train_set}_hires
if [ ! -z $max_wer ]; then
  if [ $stage -le 15 ]; then
    bad_utts_dir=${gmm_dir}_${mic}_${train_set}_bad_utts # added mic in name as this can be ihm directory where parallel mdm and sdm utts are written
    if [ ! -f $bad_utts_dir/all_info.sorted.txt ]; then
      # This stage takes a lot of time ~7hrs, so run only if file is not available already
      steps/cleanup/find_bad_utts.sh --cmd "$decode_cmd" --nj 100 data/$mic/$latgen_train_set data/lang $ali_dir $bad_utts_dir
    fi
    python local/sort_bad_utts.py --bad-utt-info-file $bad_utts_dir/all_info.sorted.txt --max-wer $max_wer --output-file $dir/wer_sorted_utts_${max_wer}wer
    utils/copy_data_dir.sh --validate-opts "--no-wav"  data/$mic/${train_set}_hires data/$mic/${train_set}_${max_wer}wer_hires
    utils/filter_scp.pl $dir/wer_sorted_utts_${max_wer}wer data/$mic/${train_set}_hires/feats.scp  > data/$mic/${train_set}_${max_wer}wer_hires/feats.scp
    utils/fix_data_dir.sh data/$mic/${train_set}_${max_wer}wer_hires
  fi
  train_data_dir=data/$mic/${train_set}_${max_wer}wer_hires
  # we don't realign again as the segment ids don't change
fi


if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/tdnn/make_configs.py $pool_opts \
    --self_repair_scale 0.00001 \
    --feat-dir $train_data_dir \
    --ivector-dir $ivector_dir \
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
    --feat.online-ivector-dir $ivector_dir \
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
         --extra-left-context 20 \
          --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) &
  done
fi
wait;
exit 0;
