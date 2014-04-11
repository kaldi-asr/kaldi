#prepare reverse lexicon and language model for backwards decoding
utils/prepare_lang.sh --reverse true data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp.reverse data/lang.reverse || exit 1;
utils/reverse_lm.sh data/local/nist_lm/lm_bg_5k.arpa.gz data/lang.reverse data/lang_test_bg_5k.reverse || exit 1;
utils/reverse_lm_test.sh data/lang_test_bg_5k data/lang_test_bg_5k.reverse || exit 1;

# normal forward decoding
utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a exp/tri2a/graph_bg5k
steps/decode_fwdbwd.sh --beam 10.0 --latbeam 4.0 --nj 8 --cmd "$decode_cmd" \
  exp/tri2a/graph_bg5k data/test_eval92 exp/tri2a/decode_eval92_bg5k_10 || exit 1;

# backward decoding
utils/mkgraph.sh --reverse data/lang_test_bg_5k.reverse exp/tri2a exp/tri2a/graph_bg5k_r
steps/decode_fwdbwd.sh --beam 10.0 --latbeam 4.0 --reverse true --nj 8 --cmd "$decode_cmd" \
  exp/tri2a/graph_bg5k_r data/test_eval92 exp/tri2a/decode_eval92_bg5k_reverse10 || exit 1;

# pingpong decoding
steps/decode_fwdbwd.sh --beam 10.0 --max-beam 20.0 --reverse true --nj 8 --cmd "$decode_cmd" \
  --first_pass exp/tri2a/decode_eval92_bg5k_10 exp/tri2a/graph_bg5k_r data/test_eval92 exp/tri2a/decode_eval92_bg5k_pingpong10 || exit 1;
steps/decode_fwdbwd.sh --beam 10.0 --max-beam 20.0 --nj 8 --cmd "$decode_cmd" \
  --first_pass exp/tri2a/decode_eval92_bg5k_reverse10 exp/tri2a/graph_bg5k data/test_eval92 exp/tri2a/decode_eval92_bg5k_pongping10 || exit 1;

# same for bigger language models (on machine with 8GB RAM, you can run the whole decoding in 3-4 min without SGE)
utils/prepare_lang.sh --reverse true data/local/dict_larger "<SPOKEN_NOISE>" data/local/lang_larger.reverse data/lang_bd.reverse || exit;
utils/reverse_lm.sh --lexicon data/local/dict_larger/lexicon.txt data/local/local_lm/3gram-mincount/lm_pr6.0.gz data/lang_bd.reverse data/lang_test_bd_tgpr.reverse || exit 1;
utils/reverse_lm_test.sh data/lang_test_bd_tgpr data/lang_test_bd_tgpr.reverse || exit 1;

utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri2a exp/tri2a/graph_bd_tgpr
steps/decode_fwdbwd.sh --beam 10.0 --latbeam 4.0 --nj 4 --cmd run.pl \
  exp/tri2a/graph_bd_tgpr data/test_eval92 exp/tri2a/decode_eval92_bdtgpr4_10 || exit 1;

utils/mkgraph.sh --reverse data/lang_test_bd_tgpr.reverse exp/tri2a exp/tri2a/graph_bd_tgpr_r
steps/decode_fwdbwd.sh --beam 10.0 --latbeam 4.0 --reverse true --nj 4 --cmd run.pl \
  exp/tri2a/graph_bd_tgpr_r data/test_eval92 exp/tri2a/decode_eval92_bdtgpr4_reverse10 || exit 1;

steps/decode_fwdbwd.sh --beam 10.0 --max-beam 20.0 --reverse true --nj 4 --cmd run.pl \
  --first_pass exp/tri2a/decode_eval92_bdtgpr4_10 exp/tri2a/graph_bd_tgpr_r data/test_eval92 \
  exp/tri2a/decode_eval92_bdtgpr4_pingpong10 || exit 1;

steps/decode_fwdbwd.sh --beam 10.0 --max-beam 20.0 --nj 4 --cmd run.pl \
  --first_pass exp/tri2a/decode_eval92_bdtgpr4_reverse10 exp/tri2a/graph_bd_tgpr data/test_eval92 \
  exp/tri2a/decode_eval92_bdtgpr4_pongping10 || exit 1;
