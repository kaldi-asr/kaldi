#!/bin/bash

# Apache2.0 
# Prepared by Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin)
#

. cmd.sh

mkdir data data/train data/eval

### Data preparation - Training data, evaluation data. Please refer http://kaldi.sourceforge.net/data_prep.html as well
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
utils/prepare_lang.sh data/local/dict.closelm "UNKNOWNGMM" data/local/lang.closelm data/lang.closelm
local/p1_format_data.sh data/lang data/lang_test data/local/lang/conv2_ears_16kwl.tg.gz
local/p1_format_data.sh data/lang.closelm data/lang_test_closelm data/local/lang/close_conv_ears_16kwl.tg.gz

### Feature extraction (training data)
mfccdir=mfcc
steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir || exit 1;
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir

utils/fix_data_dir.sh data/train

### Feature extraction (evaluation data)
steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/eval exp/make_mfcc/eval $mfccdir || exit 1;
utils/utt2spk_to_spk2utt.pl data/eval/utt2spk > data/eval/spk2utt
steps/compute_cmvn_stats.sh data/eval exp/make_mfcc/eval $mfccdir || exit 1;

utils/fix_data_dir.sh data/eval 

### We start acoustic model training here, build from HMM-GMM
### Mono phone training
steps/train_mono.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/mono0a || exit 1;
steps/align_si.sh --nj 30 --cmd "$train_cmd" data/train data/lang exp/mono0a exp/mono0a_ali

### Tri phone training
steps/train_deltas.sh --cmd "$train_cmd"  2500 20000 data/train data/lang exp/mono0a_ali exp/tri1 
steps/align_si.sh --nj 30 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
utils/mkgraph.sh data/lang_test_closelm exp/tri1 exp/tri1/graph_closelm
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri1/graph data/eval exp/tri1/decode_eval
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri1/graph_closelm data/eval exp/tri1/decode_eval_closelm

### Tri phone training (better alignment)
steps/train_deltas.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
steps/align_si.sh --nj 30 --cmd "$train_cmd" data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
utils/mkgraph.sh data/lang_test_closelm exp/tri2 exp/tri2/graph_closelm
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri2/graph data/eval exp/tri2/decode_eval
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri2/graph_closelm data/eval exp/tri2/decode_eval_closelm

### Training with LDA+MLLT feature spaces transformation
steps/train_lda_mllt.sh --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3"  2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph
utils/mkgraph.sh data/lang_test_closelm exp/tri3a exp/tri3a/graph_closelm
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri3a/graph data/eval exp/tri3a/decode_eval
steps/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri3a/graph_closelm data/eval exp/tri3a/decode_eval_closelm

### SAT (speaker adaptive training)
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;
steps/train_sat.sh  --cmd "$train_cmd" 4000 100000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;
steps/train_sat.sh  --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri3a_ali_100k exp/tri4a_20k || exit 1; 

utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph 
utils/mkgraph.sh data/lang_test_closelm exp/tri4a exp/tri4a/graph_closelm 
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri4a/graph data/eval exp/tri4a/decode_eval 
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri4a/graph_closelm data/eval exp/tri4a/decode_eval_closelm 

utils/mkgraph.sh data/lang_test exp/tri4a_20k exp/tri4a_20k/graph 
utils/mkgraph.sh data/lang_test_closelm exp/tri4a_20k exp/tri4a_20k/graph_closelm
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri4a_20k/graph data/eval exp/tri4a_20k/decode_eval
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri4a_20k/graph_closelm data/eval exp/tri4a_20k/decode_eval_closelm

### SAT (speaker adaptive training on 100K model, with better alignment)
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" data/train data/lang exp/tri4a exp/tri4a_ali_100k
steps/train_sat.sh --cmd "$train_cmd" 4000 100000 data/train data/lang exp/tri4a_ali_100k exp/tri5a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph &
utils/mkgraph.sh data/lang_test_closelm exp/tri5a exp/tri5a/graph_closelm &
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri5a/graph data/eval exp/tri5a/decode_eval &
steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" --config conf/decode.config exp/tri5a/graph_closelm data/eval exp/tri5a/decode_eval_closelm &


### Discriminative training
## (feature-space MMI + boosted MMI) 
steps/align_fmllr.sh --nj 25 --cmd "$train_cmd" data/train data/lang exp/tri5a exp/tri5a_ali_dt100k || exit 1;
steps/make_denlats.sh --nj 25 --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_dt100k --config conf/decode.config --sub-split 25 data/train data/lang exp/tri5a exp/tri5a_denlats_dt100k  || exit 1;
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 25 --cmd "$train_cmd" 800 data/train data/lang exp/tri5a_ali_dt100k exp/tri5a_dubm_dt
steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri5a_ali_dt100k exp/tri5a_dubm_dt exp/tri5a_denlats_dt100k exp/tri5a_fmmi_b0.1 || exit 1;

for n in 1 2 3 4 5 6 7 8 ; do 
steps/decode_fmmi.sh --nj 2 --cmd run.pl --iter $n --config conf/decode.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/tri5a_fmmi_b0.1/decode_eval_iter${n} & 
steps/decode_fmmi.sh --nj 2 --cmd run.pl --iter $n --config conf/decode.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/tri5a_fmmi_b0.1/decode_eval_closelm_iter${n} & 
done

## (boosted MMI only) (***remark: the lattices don't necessary re-generate again as in below two lines as exp/tri5a_ali_dt100k generated already)
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" data/train data/lang exp/tri5a exp/tri5a_ali_100k || exit 1;
steps/make_denlats.sh --nj 40 --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_100k --config conf/decode.config --sub-split 40 data/train data/lang exp/tri5a exp/tri5a_denlats_100k  || exit 1;
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 data/train data/lang exp/tri5a_{ali,denlats}_100k exp/tri5a_mmi_b0.1 || exit 1;

for n in 1 2 3 4; do 
steps/decode.sh --nj 2 --iter $n --cmd "$decode_cmd" --config conf/decode.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/tri5a_mmi_b0.1/decode_eval$n & 
steps/decode.sh --nj 2 --iter $n --cmd "$decode_cmd" --config conf/decode.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/tri5a_mmi_b0.1/decode_eval_closelm$n & 
done

## SGMM (subspace gaussian mixture model), excluding the "speaker-dependent weights"
steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 800 data/train data/lang exp/tri5a_ali_dt100k exp/ubm5a || exit 1;
steps/train_sgmm.sh  --cmd "$train_cmd" 4500 40000 data/train data/lang exp/tri5a_ali_dt100k exp/ubm5a/final.ubm exp/sgmm_5a || exit 1;

utils/mkgraph.sh data/lang_test_closelm exp/sgmm_5a exp/sgmm_5a/graph_closelm
utils/mkgraph.sh data/lang_test exp/sgmm_5a exp/sgmm_5a/graph
steps/decode_sgmm.sh --nj 2 --cmd "$decode_cmd" --transform-dir exp/tri5a/decode_eval_closelm exp/sgmm_5a/graph_closelm data/eval exp/sgmm_5a/decode_eval_closelm
steps/decode_sgmm.sh --nj 2 --cmd "$decode_cmd" --transform-dir exp/tri5a/decode_eval exp/sgmm_5a/graph data/eval exp/sgmm_5a/decode_eval

 # boosted MMI on SGMM
steps/align_sgmm.sh --nj 25 --cmd "$train_cmd" --transform-dir exp/tri5a_ali_dt100k  --use-graphs true --use-gselect true data/train data/lang exp/sgmm_5a exp/sgmm_5a_ali
steps/make_denlats_sgmm.sh --nj 25 --sub-split 25 --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_dt100k data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats
steps/train_mmi_sgmm.sh --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_dt100k --boost 0.1 data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats exp/sgmm_5a_mmi_b0.1

for n in 1 2 3 4; do
steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $n --transform-dir exp/tri5a/decode_eval_closelm data/lang_test_closelm data/eval exp/sgmm_5a/decode_eval_closelm exp/sgmm_5a_mmi_b0.1/decode_eval_closelm$n
steps/decode_sgmm_rescore.sh --cmd "$decode_cmd" --iter $n --transform-dir exp/tri5a/decode_eval data/lang_test data/eval exp/sgmm_5a/decode_eval exp/sgmm_5a_mmi_b0.1/decode_eval$n
done

### Neural Network (on top of LDA+MLLT+SAT model)
steps/train_nnet_cpu.sh --mix-up 8000 --initial-learning-rate 0.01 --final-learning-rate 0.001 --num-jobs-nnet 16 --num-hidden-layers 6 --num-parameters 8000000 --cmd "$decode_cmd" data/train data/lang exp/tri5a exp/nnet_8m_6l

 # decoding on final model for NN
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_8m_6l/decode_eval 
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_8m_6l/decode_eval_closelm 

 # better analysis, this explains why we need to have average parameters in the last ten iterations
for n in 290 280 270 260 250 240 230 220 210 200 150 100 50; do
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --iter $n --config conf/decode.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_8m_6l/decode_eval_iter${n} &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --iter $n --config conf/decode.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_8m_6l/decode_eval_closelm_iter${n} &
done

 # GPU based DNN traing, this was run on CentOS 6.4 with CUDA 5.0
 # 6 layers DNN pretrained with restricted boltzmann machine, frame level cross entropy training, sequence discriminative training with sMBR criterion
local/run_dnn.sh
 # decoding was run by CPUs
 # decoding using DNN with cross-entropy training 
dir=exp/tri5a_pretrain-dbn_dnn
steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 exp/tri5a/graph data-fmllr-tri5a/test $dir/decode || exit 1;
steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 exp/tri5a/graph_closelm data-fmllr-tri5a/test $dir/decode_closelm || exit 1;
 # decoding using DNN with sequence discriminative training (sMBR criterion)
dir=exp/tri5a_pretrain-dbn_dnn_smbr
for ITER in 1 2 3; do
 steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt 0.1 exp/tri5a/graph data-fmllr-tri5a/test $dir/decode_it${ITER} &
 steps/decode_nnet.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt 0.1 exp/tri5a/graph_closelm data-fmllr-tri5a/test $dir/decode_closelm_it${ITER} &
done


### Scoring ###
local/ext/score.sh data/eval exp/tri1/graph exp/tri1/decode_eval
local/ext/score.sh data/eval exp/tri1/graph_closelm exp/tri1/decode_eval_closelm

local/ext/score.sh data/eval exp/tri2/graph exp/tri2/decode_eval
local/ext/score.sh data/eval exp/tri2/graph_closelm exp/tri2/decode_eval_closelm

local/ext/score.sh data/eval exp/tri3a/graph exp/tri3a/decode_eval
local/ext/score.sh data/eval exp/tri3a/graph_closelm exp/tri3a/decode_eval_closelm

local/ext/score.sh data/eval exp/tri4a/graph exp/tri4a/decode_eval
local/ext/score.sh data/eval exp/tri4a/graph_closelm exp/tri4a/decode_eval_closelm

local/ext/score.sh data/eval exp/tri4a_20k/graph exp/tri4a_20k/decode_eval
local/ext/score.sh data/eval exp/tri4a_20k/graph_closelm exp/tri4a_20k/decode_eval_closelm

local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a/decode_eval
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a/decode_eval_closelm

for n in 1 2 3 4 5 6 7 8; do local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_fmmi_b0.1/decode_eval_iter$n; done
for n in 1 2 3 4 5 6 7 8; do local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_fmmi_b0.1/decode_eval_closelm_iter$n; done

local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_eval
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_eval_closelm

for n in 1 2 3 4; do local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_mmi_b0.1/decode_eval$n; done
for n in 1 2 3 4; do local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_mmi_b0.1/decode_eval_closelm$n; done

local/ext/score.sh data/eval exp/sgmm_5a/graph exp/sgmm_5a/decode_eval;
local/ext/score.sh data/eval exp/sgmm_5a/graph_closelm exp/sgmm_5a/decode_eval_closelm;

for n in 1 2 3 4; do 
 local/ext/score.sh data/eval exp/sgmm_5a/graph exp/sgmm_5a_mmi_b0.1/decode_eval$n;
 local/ext/score.sh data/eval exp/sgmm_5a/graph_closelm exp/sgmm_5a_mmi_b0.1/decode_eval_closelm$n;
done

local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_8m_6l/decode_eval
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_8m_6l/decode_eval_closelm

for n in 290 280 270 260 250 240 230 220 210 200 150 100 50; do 
  local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_8m_6l/decode_eval_iter${n}; 
  local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_8m_6l/decode_eval_closelm_iter${n}; 
done

local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_pretrain-dbn_dnn/decode
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_pretrain-dbn_dnn/decode_closelm

for ITER in 1 2 3; do
 local/ext/score.sh data/eval exp/tri5a/graph exp/tri5a_pretrain-dbn_dnn_smbr/decode_it${ITER}
 local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/tri5a_pretrain-dbn_dnn_smbr/decode_closelm_it${ITER}
done


