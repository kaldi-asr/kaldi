#!/bin/bash

. ./cmd.sh

# This is as run_sgmm2.sh but uses the "SGMM2" version of the code and
# scripts, with various improvements.

# Build a SGMM2 system on just the 100k_nodup data, on top of LDA+MLLT+SAT.
if [ ! -f exp/ubm5a/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 700 data/train_100k_nodup data/lang \
    exp/tri4a_ali_100k_nodup exp/ubm5a || exit 1;
fi

steps/train_sgmm2.sh --cmd "$train_cmd" \
  9000 30000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup \
  exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;


utils/mkgraph.sh data/lang_test exp/sgmm2_5a exp/sgmm2_5a/graph || exit 1;

steps/decode_sgmm2.sh  --cmd "$decode_cmd" --config conf/decode.config \
  --nj 30 --transform-dir exp/tri4a/decode_eval2000 \
  exp/sgmm2_5a/graph data/eval2000 exp/sgmm2_5a/decode_eval2000

 # Now discriminatively train the SGMM system on 100k_nodup data.
steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4a_ali_100k_nodup \
  --use-graphs true --use-gselect true data/train_100k_nodup data/lang exp/sgmm2_5a exp/sgmm2_5a_ali_100k_nodup

  # Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4a_ali_100k_nodup \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup exp/sgmm2_5a_mmi_b0.1

for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri4a/decode_eval2000 data/lang_test data/eval2000 exp/sgmm2_5a/decode_eval2000 \
    exp/sgmm2_5a_mmi_b0.1/decode_eval2000_it$iter &
done


(  # testing drop-frames.
 steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 --drop-frames true \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup exp/sgmm2_5a_mmi_b0.1_z

 for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri4a/decode_eval2000 data/lang_test data/eval2000 exp/sgmm2_5a/decode_eval2000 \
    exp/sgmm2_5a_mmi_b0.1_z/decode_eval2000_it$iter &
 done
 wait
)

( # testing drop-frames.
  # The same after a code speedup.
 steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 --drop-frames true \
  data/train_100k_nodup data/lang exp/sgmm2_5a_ali_100k_nodup exp/sgmm2_5a_denlats_100k_nodup exp/sgmm2_5a_mmi_b0.1_z2

 for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri4a/decode_eval2000 data/lang_test data/eval2000 exp/sgmm2_5a/decode_eval2000 \
    exp/sgmm2_5a_mmi_b0.1_z2/decode_eval2000_it$iter &
 done
 wait
)

