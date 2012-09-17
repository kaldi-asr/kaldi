#!/bin/bash

. cmd.sh


# Build a SGMM system on just the 100k_nodup data, on top of LDA+MLLT+SAT.
if [ ! -f exp/ubm5a/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 700 data/train_100k_nodup data/lang \
    exp/tri4a_ali_100k_nodup exp/ubm5a || exit 1;
fi
steps/train_sgmm.sh --cmd "$train_cmd" \
  4500 40000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup \
  exp/ubm5a/final.ubm exp/sgmm5a || exit 1;

utils/mkgraph.sh data/lang_test exp/sgmm5a exp/sgmm5a/graph || exit 1;

steps/decode_sgmm.sh  --cmd "$decode_cmd" --config conf/decode.config \
  --nj 30 --transform-dir exp/tri4a/decode_eval2000 \
  exp/sgmm5a/graph data/eval2000 exp/sgmm5a/decode_eval2000

 # Now discriminatively train the SGMM system on 100k_nodup data.
steps/align_sgmm.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4a_ali_100k_nodup \
  --use-graphs true --use-gselect true data/train_100k_nodup data/lang exp/sgmm5a exp/sgmm5a_ali_100k_nodup

  # Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm.sh --nj 30 --sub-split 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4a_ali_100k_nodup \
  data/train_100k_nodup data/lang exp/sgmm5a_ali_100k_nodup exp/sgmm5a_denlats_100k_nodup

steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri4a_ali_100k_nodup --boost 0.1 \
  data/train_100k_nodup data/lang exp/sgmm5a_ali_100k_nodup exp/sgmm5a_denlats_100k_nodup exp/sgmm5a_mmi_b0.1

for iter in 1 2 3 4; do
  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri4a/decode_eval2000 data/lang_test data/eval2000 exp/sgmm5a/decode_eval2000 \
    exp/sgmm5a_mmi_b0.1/decode_eval2000_it$iter &
done

