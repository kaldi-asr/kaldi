#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data-fbank/test
train=data-fbank/train

dev_original=data/test
train_original=data/train

gmm=exp/tri3b

stage=0
. utils/parse_options.sh


# Make the FBANK features
if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

# Run the CNN pre-training.
if [ $stage -le 1 ]; then
  dir=exp/cnn4c
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --prepend-cnn-type cnn1d --cnn-proto-opts "--patch-dim1 8 --pitch-dim 3" \
      --hid-layers 2 --learn-rate 0.008 \
      ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

# Pre-train stack of RBMs on top of the convolutional layers (4 layers, 1024 units)
if [ $stage -le 2 ]; then
  dir=exp/cnn4c_pretrain-dbn
  transf_cnn=exp/cnn4c/final.feature_transform_cnn # transform with convolutional layers
  # Train
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 4 --hid-dim 1024 --rbm-iter 20 \
    --feature-transform $transf_cnn --input-vis-type bern \
    --param-stddev-first 0.05 --param-stddev 0.05 \
    $train $dir || exit 1
fi

# Re-align using CNN
if [ $stage -le 3 ]; then
  dir=exp/cnn4c
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $train data/lang $dir ${dir}_ali || exit 1
fi

# Train the DNN optimizing cross-entropy.
if [ $stage -le 4 ]; then
  dir=exp/cnn4c_pretrain-dbn_dnn; [ ! -d $dir ] && mkdir -p $dir/log;
  ali=exp/cnn4c_ali
  feature_transform=exp/cnn4c/final.feature_transform
  feature_transform_dbn=exp/cnn4c_pretrain-dbn/final.feature_transform
  dbn=exp/cnn4c_pretrain-dbn/4.dbn
  cnn_dbn=$dir/cnn_dbn.nnet
  { # Concatenate CNN layers and DBN,
    num_components=$(nnet-info $feature_transform | grep -m1 num-components | awk '{print $2;}')
    nnet-concat "nnet-copy --remove-first-layers=$num_components $feature_transform_dbn - |" $dbn $cnn_dbn \
      2>$dir/log/concat_cnn_dbn.log || exit 1 
  }
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $cnn_dbn --hid-layers 0 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $gmm/graph $dev $dir/decode || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.2 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. For RM good acwt is 0.2
dir=exp/cnn4c_pretrain-dbn_dnn_smbr
srcdir=exp/cnn4c_pretrain-dbn_dnn
acwt=0.2

# First we generate lattices and alignments:
if [ $stage -le 4 ]; then
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

# Re-train the DNN by 6 iterations of sMBR 
if [ $stage -le 5 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1
  done 
fi

echo Success
exit 0


