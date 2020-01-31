#!/usr/bin/env bash

# This scripts tests the VTLN estimation where the system used to get the
# VTLN warps is based on delta+delta-deltas.
# see also run_vtln2.sh where the system uses LDA+MLLT instead.

. ./cmd.sh
featdir=mfcc
set -e

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd" 1800 9000 \
  data/train data/lang exp/tri2a exp/tri3d
mkdir -p data/train_vtln
cp data/train/* data/train_vtln || true
cp exp/tri3d/final.warp data/train_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/train_vtln exp/make_mfcc/train_vtln $featdir  
steps/compute_cmvn_stats.sh data/train_vtln exp/make_mfcc/train_vtln $featdir  
 utils/mkgraph.sh data/lang exp/tri3d exp/tri3d/graph
steps/decode_lvtln.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3d/graph data/test exp/tri3d/decode

mkdir -p data/test_vtln
cp data/test/* data/test_vtln || true
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
#%WER 1.84 [ 231 / 12533, 30 ins, 46 del, 155 sub ] exp/tri3b/decode/wer_4
#%WER 10.27 [ 1287 / 12533, 126 ins, 204 del, 957 sub ] exp/tri3b/decode_ug/wer_13

# With VTLN:
#%WER 2.03 [ 255 / 12533, 30 ins, 49 del, 176 sub ] exp/tri5d/decode/wer_4
#%WER 10.38 [ 1301 / 12533, 128 ins, 214 del, 959 sub ] exp/tri5d/decode_ug/wer_13

# :-( seems a bit worse.  Last time it was better.  Anyway this setup
# is too small to be really sure.
