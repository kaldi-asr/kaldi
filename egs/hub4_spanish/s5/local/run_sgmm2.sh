#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
train_nj=32
stage=0
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=80000

. ./cmd.sh
. ./path.sh


if [ $stage -le 0 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 0: Starting exp/ubm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_ubm.sh \
    --cmd "$train_cmd" $numGaussUBM \
    data/train data/langp/tri5_ali exp/tri5_ali exp/ubm5
fi

if [ $stage -le 1 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 1: Starting exp/sgmm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2.sh \
    --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
    data/train data/langp/tri5_ali exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5
fi

if [ $stage -le 2 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 2: Starting exp/sgmm5_ali/ on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
    --nj 32  --cmd "$train_cmd" --transform-dir exp/tri5_ali \
    --use-graphs true --use-gselect true \
    data/train data/langp/tri5_ali exp/sgmm5 exp/sgmm5_ali
fi

if [ $stage -le 3 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 3: Starting exp/sgmm5_denlats/ on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj 32 --sub-split 32 --num-threads 4 \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/train data/langp/tri5_ali exp/sgmm5_ali exp/sgmm5_denlats
fi

if [ $stage -le 4 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 4: Starting exp/sgmm5_mmi_b0.1/ on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mmi_sgmm2.sh \
     --cmd "$train_cmd" --drop-frames true --transform-dir exp/tri5_ali --boost 0.1 \
     data/train data/langp/tri5_ali exp/sgmm5_ali exp/sgmm5_denlats  exp/sgmm5_mmi_b0.1
fi


if [ $stage -le 5 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 5: Running decoding with SGMM2 models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/langp_test exp/sgmm5 exp/sgmm5/graph |tee exp/sgmm5/mkgraph.log

  decode=exp/sgmm5/decode_test/
  mkdir -p $decode
  steps/decode_sgmm2.sh  --beam 10 --lattice-beam 4\
    --nj 32 --cmd "$decode_cmd"\
    --transform-dir exp/tri5/decode_test \
    exp/sgmm5/graph data/eval/ ${decode} |tee ${decode}/decode.log
fi

if [ $stage -le 6 ]; then
  echo ---------------------------------------------------------------------
  echo "Stage 6: Running rescoring with SGMM2+bMMI models  on" `date`
  echo ---------------------------------------------------------------------
  for iter in 1 2 3 4; do
    decode=exp/sgmm5_mmi_b0.1/decode_test_it$iter
    mkdir -p $decode
    steps/decode_sgmm2_rescore.sh  \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_test/ \
      data/langp_test data/eval/  exp/sgmm5/decode_test $decode
  done
fi

