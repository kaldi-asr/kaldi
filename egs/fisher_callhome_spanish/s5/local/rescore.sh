#!/usr/bin/env bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0

. cmd.sh

for iter in 1 2 3 4; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri5a/decode_test data/lang data/test exp/sgmm2x_6a/decode_test_fmllr \
      exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it$iter &
done


for iter in 1 2 3 4; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri5a/decode_dev data/lang data/dev exp/sgmm2x_6a/decode_dev_fmllr \
      exp/sgmm2x_6a_mmi_b0.2/decode_dev_fmllr_it$iter &
done


for iter in 1 2 3 4; do
      steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
      --transform-dir exp/tri5a/decode_dev2 data/lang data/dev2 exp/sgmm2x_6a/decode_dev2_fmllr \
      exp/sgmm2x_6a_mmi_b0.2/decode_dev2_fmllr_it$iter &
done
