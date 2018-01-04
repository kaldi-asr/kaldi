#!/bin/bash

. ./cmd.sh

steps/align_raw_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_raw

steps/train_raw_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali_raw exp/tri3c || exit 1;

utils/mkgraph.sh data/lang exp/tri3c exp/tri3c/graph
utils/mkgraph.sh data/lang_ug exp/tri3c exp/tri3c/graph_ug

steps/decode_raw_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph data/test exp/tri3c/decode

steps/decode_raw_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph_ug data/test exp/tri3c/decode_ug

steps/decode_raw_fmllr.sh --use-normal-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph data/test exp/tri3c/decode_2fmllr

steps/decode_raw_fmllr.sh --use-normal-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri3c/graph_ug data/test exp/tri3c/decode_2fmllr_ug

steps/align_raw_fmllr.sh --nj 8 --cmd "$train_cmd" data/train data/lang exp/tri3c exp/tri3c_ali


                                        
                                                                    
if [ ! -f exp/ubm4c/final.mdl ]; then
  steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 400 data/train data/lang exp/tri3c_ali exp/ubm4c || exit 1;
fi
steps/train_sgmm2.sh  --cmd "$train_cmd" 5000 7000 data/train data/lang exp/tri3c_ali exp/ubm4c/final.ubm exp/sgmm2_4c || exit 1;

utils/mkgraph.sh data/lang exp/sgmm2_4c exp/sgmm2_4c/graph || exit 1;
utils/mkgraph.sh data/lang_ug exp/sgmm2_4c exp/sgmm2_4c/graph_ug || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3c/decode  exp/sgmm2_4c/graph data/test exp/sgmm2_4c/decode || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3c/decode_ug  exp/sgmm2_4c/graph_ug data/test exp/sgmm2_4c/decode_ug || exit 1;

steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3c/decode  exp/sgmm2_4c/graph data/test exp/sgmm2_4c/decode_fmllr || exit 1;
 

exit 0;


# (# get scaled-by-30 versions of the vecs to be used for nnet training.
#   . ./path.sh
#   mkdir -p exp/sgmm2_4c_x30
#   cat exp/sgmm2_4c/vecs.* | copy-vector ark:- ark,t:- | \
#    awk -v scale=30.0 '{printf("%s [ ", $1); for (n=3;n<NF;n++) { printf("%f ", scale*$n); } print "]"; }' > exp/sgmm2_4c_x30/vecs.1
#   mkdir -p exp/sgmm2_4c_x30/decode
#   cat exp/sgmm2_4c/decode/vecs.* | copy-vector ark:- ark,t:- | \
#    awk -v scale=30.0 '{printf("%s [ ", $1); for (n=3;n<NF;n++) { printf("%f ", scale*$n); } print "]"; }' > exp/sgmm2_4c_x30/decode/vecs.1
#   mkdir -p exp/sgmm2_4c_x30/decode_ug
#   cat exp/sgmm2_4c/decode_ug/vecs.* | copy-vector ark:- ark,t:- | \
#    awk -v scale=30.0 '{printf("%s [ ", $1); for (n=3;n<NF;n++) { printf("%f ", scale*$n); } print "]"; }' > exp/sgmm2_4c_x30/decode_ug/vecs.1
# )
# exit 0;
# ## 
# steps/decode_sgmm2.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
#   exp/sgmm2_4c.no_transform/graph data/test exp/sgmm2_4c.no_transform/decode || exit 1;

# steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
#   exp/sgmm2_4c.no_transform/graph data/test exp/sgmm2_4c.no_transform/decode_fmllr || exit 1;


