#!/bin/bash
. ./cmd.sh
[ -f path.sh ] && . ./path.sh

steps/make_denlats.sh --nj 30 --sub-split 30 --cmd "$train_cmd" \
  --transform-dir exp/tri3b_ali_train \
  data/train data/lang exp/tri3b exp/tri3b_denlats_train || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train data/lang exp/tri3b_ali_train exp/tri3b_denlats_train \
  exp/tri3b_mmi_b0.1  || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode_bg_test \
  exp/tri3b/graph_tgpr data/test exp/tri3b_mmi_b0.1/decode_bg_test

#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 30 --cmd "$train_cmd" \
  600 data/train data/lang exp/tri3b_ali_train exp/dubm3b

# Next, fMMI+MMI.
steps/train_mmi_fmmi.sh \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali_train exp/dubm3b exp/tri3b_denlats_train exp/tri3b_fmmi_a || exit 1;

for iter in 1 2 3 4; do
 steps/decode_fmmi.sh --nj 30  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode_bg_test  exp/tri3b/graph_bg data/test \
  exp/tri3b_fmmi_a/decode_bg_test_it$iter 
done

# fMMI + mmi with indirect differential.
steps/train_mmi_fmmi_indirect.sh --boost 0.1 --cmd "$train_cmd" \
data/train data/lang exp/tri3b_ali_train exp/dubm3b exp/tri3b_denlats_train \
exp/tri3b_fmmi_indirect || exit 1;

for iter in 1 2 3 4; do
 steps/decode_fmmi.sh --nj 30  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode_bg_test  exp/tri3b/graph_bg data/test \
  exp/tri3b_fmmi_indirect/decode_bg_test_it$iter 
done

 