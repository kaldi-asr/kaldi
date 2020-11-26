#!/usr/bin/env bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on ICSI+AMI to safet

# %WER 45.91 [ 8956 / 19507, 445 ins, 6660 del, 1851 sub ] exp/chain_train_icsiami/tdnn_train_icsiami/decode_safe_t_dev1_train_tl/wer_8_0.0
# %WER 14.19 [ 2768 / 19507, 281 ins, 1183 del, 1304 sub ] exp/chain_finetune/tdnn_finetune_40_100/decode_safe_t_dev1_finetune_tl/wer_8_0.0
# %WER 13.94 [ 2720 / 19507, 289 ins, 1108 del, 1323 sub ] exp/chain_finetune/tdnn_finetune_40_100_ep2/decode_safe_t_dev1_finetune_tl/wer_8_0.0

# %WER 38.17 [ 7445 / 19507, 512 ins, 5278 del, 1655 sub ] exp/chain_train_icsiami/tdnn_train_icsiami_renorm/decode_safe_t_dev1_train_tl/wer_7_0.0
# %WER 12.20 [ 2379 / 19507, 248 ins, 1000 del, 1131 sub ] exp/chain_finetune/tdnn_finetune_25_100_ep2/decode_safe_t_dev1_finetune_tl/wer_9_0.0
# %WER 11.83 [ 2308 / 19507, 225 ins, 977 del, 1106 sub ] exp/chain_finetune/tdnn_finetune_25_100_ep3/decode_safe_t_dev1_finetune_tl/wer_8_0.5

# %WER 12.61 [ 2460 / 19507, 245 ins, 1119 del, 1096 sub ] exp/chain_all/tdnn_all/decode_safe_t_dev1/wer_9_0.5
# %WER 11.70 [ 2283 / 19507, 228 ins, 964 del, 1091 sub ] exp/chain_finetune/tdnn_finetune_shared_25_100_ep2/decode_safe_t_dev1_finetune_tl/wer_8_0.5

# ./local/chain/compare_wer.sh exp/chain_finetune/tdnn_finetune_25_100_ep3/
# System                      tdnn_finetune_25_100_ep3
# WER                             11.83
# Final train prob              -0.0412
# Final valid prob              -0.0654
# Final train prob (xent)       -0.9444
# Final valid prob (xent)       -1.1019
# Parameters                     14.37M
set -e

dir=exp/chain_finetune/cnn_tdnn_finetune

src_mdl=exp/chain_all/cnn_tdnn_all//final.mdl # Input chain model
                                                   # trained on source dataset (icsi and ami).
                                                   # This model is transfered to the target domain.

src_mfcc_config=conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                     # mfcc features for ivector and DNN training
                                     # in the source domain.
src_ivec_extractor_dir=exp/nnet3_all/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.


src_tree_dir=exp/chain_all/tree_bi_all # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree

primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.

phone_lm_scales="1,10" # comma-separated list of positive integer multiplicities
                       # to apply to the different source data directories (used
                       # to give the RM data a higher weight).

set -e -o pipefail
stage=0
nj=100
train_set=train_safet
gmm=tri3
num_epochs=2

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tdnn_affix=_finetune_25_100_ep2  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
nnet3_affix=_finetune
common_egs_dir=
dropout_schedule='0,0@0.20,0.5@0.50,0'
remove_egs=true
xent_regularize=0.25
get_egs_stage=-10
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

local/nnet3/run_ivector_common_finetune.sh --stage $stage \
                                           --nj $nj \
                                           --train-set $train_set \
                                           --nnet3-affix "$nnet3_affix" \
                                           --extractor $src_ivec_extractor_dir

lores_train_data_dir=data/${train_set}_sp
train_data_dir=data/${train_set}_sp_hires
gmm_dir=exp/${gmm}_${train_set}
ali_dir=exp/${gmm}_${train_set}_ali_sp
lat_dir=exp/${gmm}_${train_set}_lats_sp
lang_dir=data/lang_nosp_test
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

if [ $stage -le 5 ]; then
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true \
  $lores_train_data_dir  $lang_dir $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz
fi

if [ $stage -le 6 ]; then
  # Set the learning-rate-factor for all transferred layers but the last output
  # layer to primary_lr_factor.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
      $src_mdl $dir/input.raw || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM with icsi,ami and safet weight $phone_lm_scales."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $src_tree_dir $lat_dir $dir || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/opensat-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  # exclude phone_LM and den.fst generation training stages
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$train_ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width 140,100,160 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 5 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $src_tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_dir $dir $dir/graph
fi

if [ $stage -le 10 ]; then

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    data/safe_t_dev1_hires $src_ivec_extractor_dir \
    exp/nnet3${nnet3_affix}/ivectors_safe_t_dev1_hires

    steps/nnet3/decode.sh --num-threads 4 --nj 20 --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_safe_t_dev1_hires \
       $dir/graph data/safe_t_dev1_hires $dir/decode_safe_t_dev1_finetune_tl || exit 1;
fi
exit 0
