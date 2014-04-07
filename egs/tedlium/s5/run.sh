#!/bin/bash

. cmd.sh
. path.sh

nj=8
decode_nj=2

# Data prep

local/data_download.sh

local/data_prep.sh

local/dict_prep.sh

utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

local/lm_prep.sh

# Feature extraction

steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train exp/make_mfcc/train mfcc
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test exp/make_mfcc/test mfcc
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test mfcc

# Train

steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang exp/mono0a

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/train data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang exp/mono0a_ali exp/tri1

utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph

steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/train data/lang exp/tri1 exp/tri1_ali

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   4000 50000 data/train data/lang exp/tri1_ali exp/tri2

utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph

steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
   exp/tri2/graph data/test exp/tri2/decode

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/train data/lang exp/tri2 exp/tri2_ali

steps/train_sat.sh --cmd "$train_cmd" \
   5000 100000 data/train data/lang exp/tri2_ali exp/tri3

utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph

steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
   exp/tri3/graph data/test exp/tri3/decode

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
   data/train data/lang exp/tri3 exp/tri3_ali

steps/make_denlats.sh --transform-dir exp/tri3_ali --nj $nj --cmd "$decode_cmd" \
   data/train data/lang exp/tri3 exp/tri3_denlats

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train data/lang exp/tri3_ali exp/tri3_denlats \
  exp/tri3_mmi_b0.1

for iter in 4; do
steps/decode.sh --transform-dir exp/tri3/decode --nj $decode_nj --cmd "$decode_cmd" --iter $iter \
   exp/tri3/graph data/test exp/tri3_mmi_b0.1/decode_it$iter
done
