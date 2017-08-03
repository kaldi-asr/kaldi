#!/bin/bash
# TODO: This is a demo of extracting ivectors for images.

# Begin configuration section.
ivector_dim=400 # dimension of the extracted i-vector
stage=0
# End configuration section.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

num_components=256
if [ $stage -le 0 ]; then
image/ivector/get_pca_transform.sh --cmd "$train_cmd" \
  --dim 40 --patch-width 8 --patch-height 8 \
  data/cifar10_train exp/diag_ubm_cifar10

image/ivector/get_pca_transform.sh --cmd "$train_cmd" \
  --dim 40 --patch-width 8 --patch-height 8 \
  data/cifar100_train exp/diag_ubm_cifar100
fi

if [ $stage -le 1 ]; then
# Train UBM and i-vector extractor.
image/ivector/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
  --nj 20 --num-threads 8 \
  data/cifar10_train $num_components \
  exp/diag_ubm_cifar10

image/ivector/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
  --nj 20 --num-threads 8 \
  data/cifar100_train $num_components \
  exp/diag_ubm_cifar100
fi

if [ $stage -le 2 ]; then
image/ivector/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd --mem 25G" data/cifar10_train \
  exp/diag_ubm_cifar10 exp/full_ubm_cifar10

image/ivector/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd --mem 25G" data/cifar100_train \
  exp/diag_ubm_cifar100 exp/full_ubm_cifar100
fi

if [ $stage -le 3 ]; then
image/ivector/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
  --ivector-dim ivector_dim \
  --num-iters 5 exp/full_ubm_cifar10/final.ubm data/cifar10_train \
  exp/extractor_cifar10

image/ivector/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
  --ivector-dim ivector_dim \
  --num-iters 5 exp/full_ubm_cifar100/final.ubm data/cifar100_train \
  exp/extractor_cifar100
fi

if [ $stage -le 4 ]; then
image/ivector/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor_cifar10 data/cifar10_train \
  exp/ivectors_cifar10_train

image/ivector/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor_cifar10 data/cifar10_test \
  exp/ivectors_cifar10_test

image/ivector/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor_cifar100 data/cifar100_train \
  exp/ivectors_cifar100_train

image/ivector/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor_cifar100 data/cifar100_test \
  exp/ivectors_cifar100_test
fi
