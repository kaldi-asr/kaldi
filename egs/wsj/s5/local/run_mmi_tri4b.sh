#!/usr/bin/env bash
. ./cmd.sh

steps/make_denlats.sh --nj 30 --sub-split 30 --cmd "$train_cmd" \
  --transform-dir exp/tri4b_ali_si284 \
  data/train_si284 data/lang exp/tri4b exp/tri4b_denlats_si284 || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 \
  exp/tri4b_mmi_b0.1  || exit 1;

steps/decode.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode_tgpr_dev93 \
  exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi_b0.1/decode_tgpr_dev93

#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 30 --cmd "$train_cmd" \
  600 data/train_si284 data/lang exp/tri4b_ali_si284 exp/dubm4b

# Next, fMMI+MMI.
steps/train_mmi_fmmi.sh \
  --boost 0.1 --cmd "$train_cmd" data/train_si284 data/lang exp/tri4b_ali_si284 exp/dubm4b exp/tri4b_denlats_si284 \
  exp/tri4b_fmmi_a || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 10  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode_tgpr_dev93  exp/tri4b/graph_tgpr data/test_dev93 \
  exp/tri4b_fmmi_a/decode_tgpr_dev93_it$iter &
done
# decode the last iter with the bd model.
for iter in 8; do
 steps/decode_fmmi.sh --nj 10  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode_bd_tgpr_dev93  exp/tri4b/graph_bd_tgpr data/test_dev93 \
  exp/tri4b_fmmi_a/decode_bd_tgpr_dev93_it$iter &
 steps/decode_fmmi.sh --nj 8  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode_bd_tgpr_eval92  exp/tri4b/graph_bd_tgpr data/test_eval92 \
  exp/tri4b_fmmi_a/decode_tgpr_eval92_it$iter &
done


# fMMI + mmi with indirect differential.
steps/train_mmi_fmmi_indirect.sh \
  --boost 0.1 --cmd "$train_cmd" data/train_si284 data/lang exp/tri4b_ali_si284 exp/dubm4b exp/tri4b_denlats_si284 \
  exp/tri4b_fmmi_indirect || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 10  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode_tgpr_dev93  exp/tri4b/graph_tgpr data/test_dev93 \
  exp/tri4b_fmmi_indirect/decode_tgpr_dev93_it$iter &
done

