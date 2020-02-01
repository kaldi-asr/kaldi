#!/usr/bin/env bash
# Copyright 2017  Radboud University (Author: Emre Yilmaz)

. ./cmd.sh
. ./path.sh

stage=0
feat_nj=10
train_nj=10
decode_nj=10
famecorpus=./corpus/ASR
. ./utils/parse_options.sh

numLeavesTri1=5000
numGaussTri1=25000
numLeavesMLLT=5000
numGaussMLLT=25000
numLeavesSAT=5000
numGaussSAT=25000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=20000

if [ -d $famecorpus ] ; then
  echo "Fame corpus present. OK."
elif [ -f ./fame.tar.gz ] ; then
  echo "Unpacking..."
  tar xzf fame.tar.gz
elif [ ! -d $famecorpus ] && [ ! -f ./fame.tar.gz ] ; then
  echo "The Fame! corpus is not present. Please register here: http://www.ru.nl/clst/datasets/ "
  echo " and download the corpus and put it at $famecorpus" && exit 1
fi

if [ $stage -le 1 ]; then
  local/fame_data_prep.sh $famecorpus || exit 1;
  local/fame_dict_prep.sh $famecorpus || exit 1;
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
  utils/format_lm.sh data/lang data/local/LM.gz data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

if [ $stage -le 2 ]; then
  # Feature extraction
  for x in train_asr devel_asr test_asr; do
      steps/make_mfcc.sh --nj $feat_nj --mfcc-config conf/mfcc_asr.conf --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc || exit 1;
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  ### Monophone
  echo "Starting monophone training."
  steps/train_mono.sh --nj $train_nj --cmd "$train_cmd" data/train_asr data/lang exp/mono || exit 1;
  echo "Mono training done."

fi

if [ $stage -le 4 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $train_nj --cmd "$train_cmd" data/train_asr data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd"  $numLeavesTri1 $numGaussTri1 data/train_asr data/lang exp/mono_ali exp/tri1 || exit 1;
  echo "Triphone training done."

fi

if [ $stage -le 5 ]; then
  ### Triphone + LDA and MLLT
  echo "Starting LDA+MLLT training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd"  data/train_asr data/lang exp/tri1 exp/tri1_ali || exit 1;
  steps/train_lda_mllt.sh  --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT data/train_asr data/lang  exp/tri1_ali exp/tri2 || exit 1;
  echo "LDA+MLLT training done."

fi

if [ $stage -le 6 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh  --nj $train_nj --cmd "$train_cmd" --use-graphs true data/train_asr data/lang exp/tri2 exp/tri2_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train_asr data/lang exp/tri2_ali exp/tri3 || exit 1;
  echo "SAT+FMLLR training done."

  echo "Decoding the development and test sets using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --skip-scoring true exp/tri3/graph data/devel_asr exp/tri3/decode_devel || exit 1;
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --skip-scoring true exp/tri3/graph data/test_asr exp/tri3/decode_test || exit 1;
  echo "SAT+FMLLR decoding done."
fi

local/dnn/run_nnet2_multisplice.sh
