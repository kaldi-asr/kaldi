#!/bin/bash

# Apache 2.0.  
#
# 2013, Hong Kong University of Science and Technology (Author: Chan Ho Yin)
#
# The decoding is the same as ../run.sh , except we use wider beam width here for comparison


. cmd.sh
. path.sh

ulimit -u 100000


steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config exp/tri5a/graph data/eval exp/tri5a/decode_wide_eval &
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config exp/tri5a/graph_closelm data/eval exp/tri5a/decode_wide_eval_closelm & 
wait

steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval exp/tri5a/graph data/eval exp/tri5a_mmi_b0.1/decode_wide_eval &
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval_closelm exp/tri5a/graph_closelm data/eval exp/tri5a_mmi_b0.1/decode_wide_eval_closelm &
wait

for n in 1 2 3 4;
do
steps/decode.sh --nj 2 --iter $n --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval exp/tri5a/graph data/eval exp/tri5a_mmi_b0.1/decode_wide_eval_$n &
steps/decode.sh --nj 2 --iter $n --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval_closelm exp/tri5a/graph_closelm data/eval exp/tri5a_mmi_b0.1/decode_wide_eval_closelm_$n &
wait
done

for n in 1 2 3 4 5 6 7 8 ;
do
steps/decode_fmmi.sh --nj 2 --cmd run.pl --iter $n --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval exp/tri5a/graph data/eval exp/tri5a_fmmi_b0.1/decode_wide_eval_iter${n} &
steps/decode_fmmi.sh --nj 2 --cmd run.pl --iter $n --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval_closelm exp/tri5a/graph_closelm data/eval exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter${n} &
wait
done

steps/decode_sgmm.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval exp/sgmm_5a/graph data/eval exp/sgmm_5a/decode_wide_eval &
steps/decode_sgmm.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval_closelm exp/sgmm_5a/graph_closelm data/eval exp/sgmm_5a/decode_wide_eval_closelm &
wait

for n in 1 2 3 4; do
  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --config conf/decode_wide.config --iter $n --transform-dir exp/tri5a/decode_wide_eval data/lang_test data/eval exp/sgmm_5a/decode_wide_eval exp/sgmm_5a_mmi_b0.1/decode_wide_eval_$n &
  steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --config conf/decode_wide.config --iter $n --transform-dir exp/tri5a/decode_wide_eval_closelm data/lang_test_closelm data/eval exp/sgmm_5a/decode_wide_eval_closelm exp/sgmm_5a_mmi_b0.1/decode_wide_eval_closelm_$n &
wait
done

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval exp/tri5a/graph data/eval exp/nnet_8m_6l/decode_wide_eval &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_wide_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_8m_6l/decode_wide_eval_closelm &
wait

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_tanh_6l/decode_wide_eval &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_tanh_6l/decode_wide_eval_closelm &
wait

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_4m_3l/decode_wide_eval &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_4m_3l/decode_wide_eval_closelm &
wait

dir=exp/tri5a_pretrain-dbn_dnn
steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn_wide.config --acwt 0.1 exp/tri5a/graph data-fmllr-tri5a/test $dir/decode_dnnwide &
steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn_wide.config --acwt 0.1 exp/tri5a/graph_closelm data-fmllr-tri5a/test $dir/decode_closelm_dnnwide &
wait
# decoding using DNN with sequence discriminative training (sMBR criterion)
dir=exp/tri5a_pretrain-dbn_dnn_smbr
for ITER in 1 2; do
  steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn_wide.config --nnet $dir/${ITER}.nnet --acwt 0.1 exp/tri5a/graph data-fmllr-tri5a/test $dir/decode_it${ITER}_dnnwide &
  steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn_wide.config --nnet $dir/${ITER}.nnet --acwt 0.1 exp/tri5a/graph_closelm data-fmllr-tri5a/test $dir/decode_closelm_it${ITER}_dnnwide &
wait
done


local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_wide_eval_1
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_wide_eval_2
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_wide_eval_3
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_wide_eval_4
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter1
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter2
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter3
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter4
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter5
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter6
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter7
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_wide_eval_iter8
local/ext/score.sh data/eval exp/tri5a/graph exp/sgmm_5a/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/sgmm_5a_mmi_b0.1/decode_wide_eval_1
local/ext/score.sh data/eval exp/tri5a/graph exp/sgmm_5a_mmi_b0.1/decode_wide_eval_2
local/ext/score.sh data/eval exp/tri5a/graph exp/sgmm_5a_mmi_b0.1/decode_wide_eval_3
local/ext/score.sh data/eval exp/tri5a/graph exp/sgmm_5a_mmi_b0.1/decode_wide_eval_4
local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_8m_6l/decode_nnwide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_tanh_6l/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_4m_3l/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_pretrain-dbn_dnn/decode_dnnwide
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_pretrain-dbn_dnn_smbr/decode_it1_dnnwide
local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_pretrain-dbn_dnn_smbr/decode_it2_dnnwide

local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a/decode_wide_eval_closelm                          # LDA+MLLT+SAT
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_wide_eval_closelm		 
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_wide_eval_closelm_1
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_wide_eval_closelm_2
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_wide_eval_closelm_3
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_wide_eval_closelm_4               # bMMI on tri5a
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter1
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter2
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter3
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter4
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter5
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter6
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter7
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_wide_eval_closelm_iter8          # fMMI+bMMI on tri5a
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/sgmm_5a/decode_wide_eval_closelm
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/sgmm_5a_mmi_b0.1/decode_wide_eval_closelm_1
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/sgmm_5a_mmi_b0.1/decode_wide_eval_closelm_2
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/sgmm_5a_mmi_b0.1/decode_wide_eval_closelm_3
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/sgmm_5a_mmi_b0.1/decode_wide_eval_closelm_4             # sgmm+bMMI 
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_8m_6l/decode_nnwide_eval_closelm                   # nnet 6 hidden layers (983 neurons)
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_tanh_6l/decode_wide_eval_closelm                   # nnet2 6 hidden layers (1024 neurons)
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_4m_3l/decode_wide_eval_closelm                     # nnet 3 hidden layers (823 neurons)
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_pretrain-dbn_dnn/decode_closelm_dnnwide           # pretrained 6 hidden layers RBM DNN
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_pretrain-dbn_dnn_smbr/decode_closelm_it1_dnnwide 
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_pretrain-dbn_dnn_smbr/decode_closelm_it2_dnnwide  # state level minimum bayes risk DNN

# grep CER exp/*/decode*wide*/cer_10 >> RESULTS

