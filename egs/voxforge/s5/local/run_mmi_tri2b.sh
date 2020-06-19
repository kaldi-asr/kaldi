#!/usr/bin/env bash

. ./cmd.sh

# Train and test MMI (and boosted MMI) on tri2b system.
steps/make_denlats.sh --sub-split 20 --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b exp/tri2b_denlats_si84 || exit 1;

# train the basic MMI system.
steps/train_mmi.sh --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 \
  exp/tri2b_denlats_si84 exp/tri2b_mmi  || exit 1;
for iter in 3 4; do
  steps/decode_si.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi/decode_tgpr_dev93_it$iter &
  steps/decode_si.sh --nj 8 --cmd "$decode_cmd" --iter $iter \
     exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi/decode_tgpr_eval92_it$iter &
done

# MMI with 0.1 boosting factor.
steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 \
  exp/tri2b_mmi_b0.1  || exit 1;

for iter in 3 4; do
  steps/decode_si.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
    exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_mmi_b0.1/decode_tgpr_dev93_it$iter &
  steps/decode_si.sh --nj 8 --cmd "$decode_cmd" --iter $iter \
     exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_b0.1/decode_tgpr_eval92_it$iter &
done


# Train a UBM with 400 components, for fMMI.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 10 --cmd "$train_cmd" \
  400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b

 steps/train_mmi_fmmi.sh --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_b0.1

 for iter in `seq 3 8`; do
   steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
     exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1/decode_tgpr_dev93_it$iter &
 done

 steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_b0.1_lr0.005 || exit 1;
 for iter in `seq 3 8`; do
   steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
     exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_b0.1_lr0.005/decode_tgpr_dev93_it$iter &
 done

 steps/train_mmi_fmmi_indirect.sh --boost 0.1 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/tri2b_ali_si84 exp/dubm2b exp/tri2b_denlats_si84 \
   exp/tri2b_fmmi_indirect_b0.1
 for iter in `seq 3 8`; do
   steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
      exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b_fmmi_indirect_b0.1/decode_tgpr_dev93_it$iter &
 done
