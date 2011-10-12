

train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"

steps/make_denlats_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84
steps/train_lda_etc_mmi.sh --num-jobs 10  --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi/decode_tgpr_eval92

# Larger beam:
steps/make_denlats_lda_etc.sh --num-jobs 10 --cmd "$train_cmd" \
  --beam 16.0 --lattice-beam 8.0 \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84b
steps/train_lda_etc_mmi.sh --num-jobs 10  --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84b exp/tri2b exp/tri2b_mmib
scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmib/decode_tgpr_eval92

steps/train_lda_etc_mmi.sh --num-jobs 10 --boost 0.1 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_mmi_b0.1

scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_mllt.sh exp/tri2b/graph_tgpr data/test_eval92 exp/tri2b_mmi_b0.1/decode_tgpr_eval92

#[ after training tri4b ]
steps/align_lda_mllt_sat.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b exp/tri4b_ali_si284
steps/make_denlats_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284

steps/train_lda_etc_mmi.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b_denlats_si284 exp/tri4b exp/tri4b_mmi

scripts/decode.sh --cmd "$decode_cmd" steps/decode_lda_etc.sh exp/tri4b/graph_tgpr data/test_dev93 exp/tri4b_mmi/decode_tgpr_dev93 exp/tri4b/decode_tgpr_dev93





