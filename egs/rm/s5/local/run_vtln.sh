#!/bin/bash

# This scripts tests the VTLN estimation where the system used to get the
# VTLN warps is based on delta+delta-deltas.
# see also run_vtln2.sh where the system uses LDA+MLLT instead.

. cmd.sh
featdir=mfcc

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd" 1800 9000 \
  data/train data/lang exp/tri2a exp/tri3d
cp -rT data/train data/train_vtln
cp exp/tri3d/final.warp data/train_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/train_vtln exp/make_mfcc/train_vtln $featdir  
steps/compute_cmvn_stats.sh data/train_vtln exp/make_mfcc/train_vtln $featdir  
 utils/mkgraph.sh data/lang exp/tri3d exp/tri3d/graph
steps/decode_lvtln.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3d/graph data/test exp/tri3d/decode

cp -rT data/test data/test_vtln
cp exp/tri3d/decode/final.warp data/test_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/test_vtln exp/make_mfcc/test_vtln $featdir  
steps/compute_cmvn_stats.sh data/test_vtln exp/make_mfcc/test_vtln $featdir  

(
 steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  1800 9000 data/train_vtln data/lang exp/tri3d exp/tri4d
 utils/mkgraph.sh data/lang exp/tri4d exp/tri4d/graph

 steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
    exp/tri4d/graph data/test_vtln exp/tri4d/decode

 steps/train_sat.sh 1800 9000 data/train_vtln data/lang exp/tri4d exp/tri5d
 utils/mkgraph.sh data/lang exp/tri5d exp/tri5d/graph 
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5d/graph data/test_vtln exp/tri5d/decode 

 utils/mkgraph.sh data/lang_ug exp/tri5d exp/tri5d/graph_ug
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5d/graph_ug data/test_vtln exp/tri5d/decode_ug
)

# Baseline with no VTLN:
#%WER 2.06 [ 258 / 12533, 37 ins, 47 del, 174 sub ] exp/tri3b/decode/wer_4
#%WER 10.17 [ 1275 / 12533, 123 ins, 191 del, 961 sub ] exp/tri3b/decode_ug/wer_13

# With VTLN:
#%WER 1.99 [ 250 / 12533, 18 ins, 70 del, 162 sub ] exp/tri5d/decode/wer_10
#%WER 9.89 [ 1239 / 12533, 119 ins, 203 del, 917 sub ] exp/tri5d/decode_ug/wer_13

