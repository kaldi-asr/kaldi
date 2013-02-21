#!/bin/bash

# This script is invoked from ../run.sh
# It contains some SGMM-related scripts that I am breaking out of the main run.sh for clarity.

. cmd.sh



# SGMM system on the si284 data
steps/train_ubm.sh --cmd "$train_cmd" \
    600 data/train_si284 data/lang exp/tri5_ali_si284 exp/ubm6 || exit 1;

steps/train_sgmm2.sh --cmd "$train_cmd" \
    11000 25000 data/train_si284 data/lang exp/tri5_ali_si284 \
    exp/ubm6/final.ubm exp/sgmm6 || exit 1;

(
    utils/mkgraph.sh data/lang_test_tgpr exp/sgmm6 exp/sgmm6/graph_tgpr
    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_tgpr_dev93 \
	exp/sgmm6/graph_tgpr data/test_dev93 exp/sgmm6/decode_tgpr_dev93
    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_tgpr_eval92 \
	exp/sgmm6/graph_tgpr data/test_eval92 exp/sgmm6/decode_tgpr_eval92

    utils/mkgraph.sh data/lang_test_bd_tgpr exp/sgmm6 exp/sgmm6/graph_bd_tgpr || exit 1;
    steps/decode_sgmm2.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_bd_tgpr_dev93 \
	exp/sgmm6/graph_bd_tgpr data/test_dev93 exp/sgmm6/decode_bd_tgpr_dev93
    steps/decode_sgmm2.sh --nj 8 --cmd "$decode_cmd" --transform-dir exp/tri5/decode_bd_tgpr_eval92 \
	exp/sgmm6/graph_bd_tgpr data/test_eval92 exp/sgmm6/decode_bd_tgpr_eval92
) &

steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri5_ali_si284 \
    --use-graphs true --use-gselect true data/train_si284 data/lang exp/sgmm6 exp/sgmm6_ali_si284 

steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --cmd "$decode_cmd" --transform-dir exp/tri5_ali_si284 \
    data/train_si284 data/lang exp/sgmm6_ali_si284 exp/sgmm6_denlats_si284

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri5_ali_si284 --boost 0.1 \
    data/train_si284 data/lang exp/sgmm6_ali_si284 exp/sgmm6_denlats_si284 exp/sgmm6_mmi_b0.1

for iter in 1 2 3 4; do
    for test in eval92; do # dev93
	steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
            --transform-dir exp/tri5/decode_bd_tgpr_${test} data/lang_test_bd_fg data/test_${test} exp/sgmm6/decode_bd_tgpr_${test} \
            exp/sgmm6_mmi_b0.1/decode_bd_tgpr_${test}_it$iter &
    done
done
) &

wait
