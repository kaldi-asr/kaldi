utils/mkgraph.sh data/lang_test_bg_5k exp/tri2a exp/tri2a/graph_bg5k
steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri2a/graph_bg5k data/test_eval92 exp/tri2a/decode_eval92_bg5k || exit 1;

#prepare reverse lexicon and language model for backwards decoding
utils/prepare_lang.sh --reverse true data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp.reverse data/lang.reverse || exit 1;
local/reverse/wsj_reverse_lm.sh bg_5k || exit 1;
local/reverse/wsj_test_reverse_lm.sh data/lang_test_bg_5k data/lang_test_bg_5k.reverse || exit 1;

utils/mkgraph.sh --reverse data/lang_test_bg_5k.reverse exp/tri2a exp/tri2a/graph_bg5kr
steps/decode_fwdbwd.sh --reverse true --nj 8 --cmd "$decode_cmd" \
  exp/tri2a/graph_bg5kr data/test_eval92 exp/tri2a/decode_eval92_bg5k_reverse || exit 1;

steps/decode_fwdbwd.sh --reverse true --nj 8 --cmd "$decode_cmd" \
  --first_pass exp/tri2a/decode_eval92_bg5k exp/tri2a/graph_bg5kr data/test_eval92 exp/tri2a/decode_eval92_bg5k_pingpong || exit 1;

# same for bigger language models
utils/prepare_lang.sh --reverse true data/local/dict_larger "<SPOKEN_NOISE>" data/local/lang_larger.reverse data/lang_bd.reverse || exit;
local/reverse/wsj_reverse_lm.sh --lm data/local/local_lm/3gram-mincount/lm_pr6.0.gz --datadir data/lang_bd.reverse --lexicon data/local/dict_larger/lexicon.txt bd_tgpr
local/reverse/wsj_test_reverse_lm.sh data/lang_test_bd_tgpr data/lang_test_bd_tgpr.reverse || exit 1;
utils/mkgraph.sh --reverse data/lang_test_bd_tgpr.reverse exp/tri2a exp/tri2a/graph_bd_tgpr_r
