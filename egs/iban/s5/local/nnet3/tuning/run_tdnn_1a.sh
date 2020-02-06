#!/usr/bin/env bash

# cat exp/nnet3/tdnn_1a/decode_dev/scoring_kaldi/best_wer
# %WER 17.34 [ 1908 / 11006, 257 ins, 303 del, 1348 sub ] exp/nnet3/tdnn_1a/decode_dev/wer_12_0.0
# cat exp/nnet3/tdnn_1a/decode_dev.rescored/scoring_kaldi/best_wer
# %WER 15.57 [ 1714 / 11006, 226 ins, 297 del, 1191 sub ] exp/nnet3/tdnn_1a/decode_dev.rescored/wer_13_0.0

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_set=train
train_stage=-10
affix=1a
gmm=tri3b
common_egs_dir=
remove_egs=true

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm || exit 1;

ali_dir=exp/${gmm}_ali_${train_set}_sp
dir=exp/nnet3/tdnn_$affix
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3/ivectors_${train_set}_sp_hires

if [ $stage -le 8 ]; then
  echo "$0: creating neural net configs";

  num_targets=$(tree-info $ali_dir/tree | grep num-pdfs | awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=50 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(-1,0,1) dim=256
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=256
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=256
  relu-renorm-layer name=tdnn4 input=Append(-1,0,1) dim=256
  relu-renorm-layer name=tdnn5 input=Append(-1,0,1) dim=256
  relu-renorm-layer name=tdnn6 dim=256

  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi



if [ $stage -le 9 ]; then

  steps/nnet3/train_dnn.py --stage $train_stage \
    --cmd="$decode_cmd" \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.num-epochs 3 \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.optimization.initial-effective-lrate 0.005 \
    --trainer.optimization.final-effective-lrate 0.0005 \
    --trainer.samples-per-iter 120000 \
    --egs.dir "$common_egs_dir" \
    --cleanup.preserve-model-interval 10 \
    --cleanup.remove-egs=$remove_egs \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang data/lang \
    --dir=$dir  || exit 1;
fi


if [ $stage -le 10 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  graph_dir=exp/tri3b/graph
  # use already-built graphs.
    steps/nnet3/decode.sh --nj 6 --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_dev_hires --iter final \
       $graph_dir data/dev_hires $dir/decode_dev || exit 1;
fi

if [ $stage -le 11 ]; then
   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
     data/lang_test/ data/lang_big/ data/dev_hires \
    ${dir}/decode_dev ${dir}/decode_dev.rescored
fi

exit 0;

