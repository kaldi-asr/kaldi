#!/bin/bash

. cmd.sh
featdir=mfcc

# train linear vtln


steps/train_lvtln.sh --stage 1000 --cmd "$train_cmd" 1800 9000 \
   data/train data/lang exp/tri2b exp/tri3e

cp -rT data/train data/train_vtln
cp exp/tri3e/final.warp data/train_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/train_vtln exp/make_mfcc/train_vtln $featdir  
steps/compute_cmvn_stats.sh data/train_vtln exp/make_mfcc/train_vtln $featdir  
 utils/mkgraph.sh data/lang exp/tri3e exp/tri3e/graph
steps/decode_lvtln.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3e/graph data/test exp/tri3e/decode

cp -rT data/test data/test_vtln
cp exp/tri3e/decode/final.warp data/test_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/test_vtln exp/make_mfcc/test_vtln $featdir  
steps/compute_cmvn_stats.sh data/test_vtln exp/make_mfcc/test_vtln $featdir  

(
 steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  1800 9000 data/train_vtln data/lang exp/tri3e exp/tri4e
 utils/mkgraph.sh data/lang exp/tri4e exp/tri4e/graph

 steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
    exp/tri4e/graph data/test_vtln exp/tri4e/decode

 steps/train_sat.sh 1800 9000 data/train_vtln data/lang exp/tri4e exp/tri5e
 utils/mkgraph.sh data/lang exp/tri5e exp/tri5e/graph 
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5e/graph data/test_vtln exp/tri5e/decode 

 utils/mkgraph.sh data/lang_ug exp/tri5e exp/tri5e/graph_ug
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5e/graph_ug data/test_vtln exp/tri5e/decode_ug
)

# Baseline with no VTLN:
#%WER 2.06 [ 258 / 12533, 37 ins, 47 del, 174 sub ] exp/tri3b/decode/wer_4
#%WER 10.17 [ 1275 / 12533, 123 ins, 191 del, 961 sub ] exp/tri3b/decode_ug/wer_13

