#prepare reverse lexicon and language model for backwards decoding
utils/prepare_lang.sh --reverse true data/local/dict "<UNK>" data/local/lang.reverse data/lang.reverse
utils/reverse_lm.sh --lexicon data/local/lang.reverse/lexicon.txt data/local/lm/3gram-mincount/lm_unpruned.gz data/lang.reverse data/lang_test.reverse
utils/reverse_lm_test.sh data/lang_test data/lang_test.reverse

# normal forward decoding
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode_fwdbwd.sh --beam 11.0 --lattice-beam 4.0 --nj 30 --cmd "$decode_cmd" exp/tri2/graph data/eval2000 exp/tri2/decode_eval2000_11

# backwards decoding
utils/mkgraph.sh --reverse data/lang_test.reverse exp/tri2 exp/tri2/graph_r
steps/decode_fwdbwd.sh --beam 11.0 --lattice-beam 4.0 --reverse true --nj 30 --cmd "$decode_cmd" \
  exp/tri2/graph_r data/eval2000 exp/tri2/decode_eval2000_reverse11

# pingpong decoding
steps/decode_fwdbwd.sh --beam 11.0 --max-beam 22.0 --reverse true --nj 30 --cmd "$decode_cmd" \
  --first_pass exp/tri2/decode_eval2000_11 exp/tri2/graph_r data/eval2000 \
  exp/tri2/decode_eval2000_pingpong11

steps/decode_fwdbwd.sh --beam 11.0 --max-beam 22.0 --nj 30 --cmd "$decode_cmd" \
  --first_pass exp/tri2/decode_eval2000_reverse11 exp/tri2/graph data/test_eval2000 \
  exp/tri2/decode_eval2000_pongping11
