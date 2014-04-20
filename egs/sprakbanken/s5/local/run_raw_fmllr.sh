#!/bin/bash


steps/align_raw_fmllr.sh --nj 10 --cmd "$train_cmd" --use-graphs true \
    data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84_raw

steps/train_raw_sat.sh --cmd "$train_cmd" \
   2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84_raw exp/tri3c || exit 1;


mfccdir=mfcc
for x in test_eval92 test_eval93 test_dev93 ; do
  y=${x}_utt
  cp -rT data/$x data/$y
  cat data/$x/utt2spk | awk '{print $1, $1;}' > data/$y/utt2spk;
  cp data/$y/utt2spk data/$y/spk2utt;
  steps/compute_cmvn_stats.sh data/$y exp/make_mfcc/$y $mfccdir || exit 1; 
done

(
utils/mkgraph.sh data/lang_test_tgpr exp/tri3c exp/tri3c/graph_tgpr || exit 1;
steps/decode_raw_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_dev93 exp/tri3c/decode_tgpr_dev93 || exit 1;
steps/decode_raw_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_eval92 exp/tri3c/decode_tgpr_eval92 || exit 1;

steps/decode_raw_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_dev93_utt exp/tri3c/decode_tgpr_dev93_utt || exit 1;
steps/decode_raw_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_eval92_utt exp/tri3c/decode_tgpr_eval92_utt || exit 1;

steps/decode_raw_fmllr.sh --use-normal-fmllr true --nj 10 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_dev93 exp/tri3c/decode_tgpr_dev93_2fmllr || exit 1;
steps/decode_raw_fmllr.sh --use-normal-fmllr true --nj 8 --cmd "$decode_cmd" \
  exp/tri3c/graph_tgpr data/test_eval92 exp/tri3c/decode_tgpr_eval92_2fmllr || exit 1;
)&

(
utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri3c exp/tri3c/graph_bd_tgpr || exit 1; 

steps/decode_raw_fmllr.sh --cmd "$decode_cmd" --nj 8 exp/tri3c/graph_bd_tgpr \
    data/test_eval92 exp/tri3c/decode_bd_tgpr_eval92 
 steps/decode_raw_fmllr.sh --cmd "$decode_cmd" --nj 10 exp/tri3c/graph_bd_tgpr \
   data/test_dev93 exp/tri3c/decode_bd_tgpr_dev93 
)&

steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_si284 data/lang exp/tri3c exp/tri3c_ali_si284 || exit 1;


steps/train_raw_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train_si284 data/lang exp/tri3c_ali_si284 exp/tri4d || exit 1;
(
 utils/mkgraph.sh data/lang_test_tgpr exp/tri4d exp/tri4d/graph_tgpr || exit 1;
 steps/decode_raw_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri4d/graph_tgpr data/test_dev93 exp/tri4d/decode_tgpr_dev93 || exit 1;
 steps/decode_raw_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri4d/graph_tgpr data/test_eval92 exp/tri4d/decode_tgpr_eval92 || exit 1;
) & 


wait


#for x in exp/tri3{b,c}/decode_tgpr*; do grep WER $x/wer_* | utils/best_wer.sh ; done

