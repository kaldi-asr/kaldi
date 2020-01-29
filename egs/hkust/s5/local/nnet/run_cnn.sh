#!/usr/bin/env bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data_fbank/dev
train=data_fbank/train

dev_original=data/dev
train_original=data/train

gmm=exp/tri5a

stage=0
. utils/parse_options.sh || exit 1;


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
  dir=exp/cnn5c
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --network-type cnn1d --cnn-proto-opts "--patch-dim1 7 --pitch-dim 3" \
      --hid-layers 2 --learn-rate 0.008 \
      ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode with the trigram language model.
  steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode || exit 1;
fi

# Pre-train stack of RBMs on top of the convolutional layers (2 layers, 2000 units)
if [ $stage -le 2 ]; then
  dir=exp/cnn5c_pretrain-dbn
  transf_cnn=exp/cnn5c/final.feature_transform_cnn # transform with convolutional layers
  # Train
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 2 --hid-dim 2000 --rbm-iter 1 \
    --feature-transform $transf_cnn --input-vis-type bern \
    --param-stddev-first 0.05 --param-stddev 0.05 \
    $train $dir || exit 1
fi

# Re-align using CNN
if [ $stage -le 3 ]; then
  dir=exp/cnn5c
  steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    $train data/lang $dir ${dir}_ali || exit 1
fi

# Train the DNN optimizing cross-entropy.
if [ $stage -le 4 ]; then
  dir=exp/cnn5c_pretrain-dbn_dnn; [ ! -d $dir ] && mkdir -p $dir/log;
  ali=exp/cnn5c_ali
  feature_transform=exp/cnn5c/final.feature_transform
  feature_transform_dbn=exp/cnn5c_pretrain-dbn/final.feature_transform
  dbn=exp/cnn5c_pretrain-dbn/2.dbn
  cnn_dbn=$dir/cnn_dbn.nnet
  { # Concatenate CNN layers and DBN,
    num_components=$(nnet-info $feature_transform | grep -m1 num-components | awk '{print $2;}')
    nnet-concat "nnet-copy --remove-first-components=$num_components $feature_transform_dbn - |" $dbn $cnn_dbn \
      2>$dir/log/concat_cnn_dbn.log || exit 1 
  }
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $cnn_dbn --hid-layers 0 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode with the trigram language model.
  steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. For RM good acwt is 0.2 (For WSJ maybe 0.1)
dir=exp/cnn5c_pretrain-dbn_dnn_smbr
srcdir=exp/cnn5c_pretrain-dbn_dnn
acwt=0.1

# First we generate lattices and alignments:
if [ $stage -le 5 ]; then
  steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

# Re-train the DNN by 2 iterations of sMBR 
if [ $stage -le 6 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2; do
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config --acwt $acwt --nnet $dir/${ITER}.nnet \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1;
  done 
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/cnn5c_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/cnn5c_pretrain-dbn_dnn_smbr

if [ $stage -le 7 ]; then
  steps/nnet/align.sh --nj 10 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi
    
if [ $stage -le 8 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config --acwt $acwt --nnet $dir/${ITER}.nnet \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1;
  done 
fi

echo Success
exit 0


