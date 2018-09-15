#!/bin/bash

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

stage=0
feat_nj=10
train_nj=10
decode_nj=10
famecorpus=./corpus

if [ -d $famecorpus ] ; then
  echo "Fame corpus present. OK."
elif [ -f ./fame.tar.gz ] ; then
  echo "Unpacking..."
  tar xzf fame.tar.gz
elif [ ! -d $famecorpus ] && [ ! -f ./fame.tar.gz ] ; then
  echo "The Fame! corpus is not present. Please register here: http://www.ru.nl/clst/datasets/ "
  echo " and download the corpus and put it at $famecorpus" && exit 1
fi

numLeavesTri1=5000
numGaussTri1=25000
numLeavesMLLT=5000
numGaussMLLT=25000
numLeavesSAT=5000
numGaussSAT=25000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=20000

if [ $stage -le 1 ]; then
  local/fame_data_prep.sh $famecorpus || exit 1;
  local/fame_dict_prep.sh $famecorpus || exit 1;
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
  utils/format_lm.sh data/lang data/local/LM.gz data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

if [ $stage -le 2 ]; then
  # Feature extraction
  for x in train devel test; do
      steps/make_mfcc.sh --nj $feat_nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  ### Monophone
  echo "Starting monophone training."
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/mono || exit 1;
  echo "Mono training done."

  echo "Decoding the development and test sets using monophone models."
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/mono/graph data/devel exp/mono/decode_devel || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode_test || exit 1;
  echo "Monophone decoding done."
fi


if [ $stage -le 4 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd"  $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
  echo "Triphone training done."

  echo "Decoding the development and test sets using triphone models."
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri1/graph data/devel exp/tri1/decode_devel || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode_test || exit 1;
  echo "Triphone decoding done."
fi

if [ $stage -le 5 ]; then
  ### Triphone + LDA and MLLT
  echo "Starting LDA+MLLT training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd"  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_lda_mllt.sh  --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT data/train data/lang  exp/tri1_ali exp/tri2 || exit 1;
  echo "LDA+MLLT training done."

  echo "Decoding the development and test sets using LDA+MLLT models."
  utils/mkgraph.sh data/lang_test  exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph data/devel exp/tri2/decode_devel || exit 1;
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2/decode_test || exit 1;
  echo "LDA+MLLT decoding done."
fi


if [ $stage -le 6 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3 || exit 1;
  echo "SAT+FMLLR training done."

  echo "Decoding the development and test sets using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri3/graph data/devel exp/tri3/decode_devel || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri3/graph data/test exp/tri3/decode_test || exit 1;
  echo "SAT+FMLLR decoding done."
fi


if [ $stage -le 7 ]; then
  echo "Starting SGMM training."
  steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" data/train data/lang exp/tri3 exp/tri3_ali || exit 1;
  steps/train_ubm.sh --cmd "$train_cmd" $numGaussUBM data/train data/lang exp/tri3_ali exp/ubm || exit 1;
  steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM data/train data/lang exp/tri3_ali exp/ubm/final.ubm exp/sgmm2 || exit 1;
  echo "SGMM training done."

  echo "Decoding the development and test sets using SGMM models"
  utils/mkgraph.sh data/lang_test exp/sgmm2 exp/sgmm2/graph || exit 1;
  steps/decode_sgmm2.sh --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri3/decode_devel exp/sgmm2/graph data/devel exp/sgmm2/decode_devel || exit 1;
  steps/decode_sgmm2.sh --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri3/decode_test exp/sgmm2/graph data/test exp/sgmm2/decode_test || exit 1;
  echo "SGMM decoding done."
fi

if [ $stage -le 8 ]; then
  echo "Starting DNN training and decoding."
  local/nnet/run_dnn.sh || exit 1;
  local/nnet/run_dnn_fbank.sh || exit 1;
fi

#score
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
