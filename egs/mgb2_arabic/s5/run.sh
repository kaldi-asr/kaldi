#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU
#               2017-19 Vimal Manohar
# Apache 2.0

stage=-1

# preference on how to process xml file [python, xml]
process_xml="python"

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

# See README for instructions.
# This script assumes that you are already familiar with Kaldi recipes.
# This script assumes that you have downloaded the corpus and lexicon as 
# mentioned in the README.

# TO DO: you will need to choose the size of training set you want.
# Here we select according to an upper threshhold on Matching Error
# Rate from the lightly supervised alignment.  When using all the
# training shows, this will give you training data speech segments of
# approximate lengths listed below:


set -e -o pipefail -u
#FILTER OUT SEGMENTS BASED ON MER (Match Error Rate)

mer=80  

# Location of lexicon
# Download from https://github.com/qcri/ArabicASRChallenge2016/blob/master/lexicon/ar-ar_grapheme_lexicon
LEXICON=ar-ar_grapheme_lexicon

nj=100  # split training into how many jobs?
nDecodeJobs=80

##########################################################
#
#  Recipe
#
##########################################################


#1) Data preparation

if [ $stage -le 0 ]; then
  local/mgb_extract_data.sh DB
fi

if [ $stage -le 1 ]; then
  #DATA PREPARATION
  echo "Preparing training data"
  local/mgb_data_prep.sh DB $mer $process_xml
fi

if [ $stage -le 2 ]; then
  #LEXICON PREPARATION: The lexicon is also provided
  echo "Preparing dictionary"
  local/graphgeme_mgb_prep_dict.sh $LEXICON
fi

# Using the training data transcript for building the language model
LM_TEXT=DB/train/lm_text/lm_text_clean_bw

if [ $stage -le 3 ]; then
  #LM TRAINING: Using the training set transcript text for language modelling
  echo "Training n-gram language model"
  local/mgb_train_lms.sh $mer
  local/mgb_train_lms_extra.sh $LM_TEXT $mer

  # Uncomment if you want to use pocolm for language modeling 
  #local/mgb_train_lms_extra_pocolm.sh $LM_TEXT $mer
fi

if [ $stage -le 4 ]; then
  #L Compilation
  echo "Preparing lang dir"
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
fi

if [ $stage -le 5 ]; then
  #G compilation
  local/mgb_format_data.sh --lang-test data/lang_test \
    --arpa-lm data/local/lm_mer80/3gram-mincount/lm_unpruned.gz
  utils/build_const_arpa_lm.sh data/local/lm_large_mer80/4gram-mincount/lm_unpruned.gz \
    data/lang_test data/lang_test_fg
fi

# Uncomment if you want to use pocolm for language modeling 
#if [ $stage -le 6 ]; then
#  local/mgb_format_data.sh --lang-test data/lang_poco_test \
#    --arpa-lm data/local/pocolm/data/arpa/4gram_small.arpa.gz
#  utils/build_const_arpa_lm.sh data/local/pocolm/data/arpa/4gram_big.arpa.gz \
#    data/lang_poco_test data/lang_poco_test_fg
#fi

if [ $stage -le 7 ]; then
  #Calculating mfcc features
  mfccdir=mfcc
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mgb2_arabic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  echo "Computing features"
  for x in train_mer$mer train_mer${mer}_subset500 dev_non_overlap dev_overlap ; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$x \
      exp/mer$mer/make_mfcc/$x/log $mfccdir
    steps/compute_cmvn_stats.sh data/$x \
      exp/mer$mer/make_mfcc/$x/log $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 8 ]; then
  #Taking 10k segments for faster training
  utils/subset_data_dir.sh data/train_mer${mer}_subset500 10000 data/train_mer${mer}_subset500_10k 
fi

if [ $stage -le 9 ]; then
  #Monophone training
  steps/train_mono.sh --nj 80 --cmd "$train_cmd" \
    data/train_mer${mer}_subset500_10k data/lang exp/mer$mer/mono 
fi

if [ $stage -le 10 ]; then
  #Monophone alignment
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_mer${mer}_subset500 data/lang exp/mer$mer/mono exp/mer$mer/mono_ali 

  #tri1 [First triphone pass]
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train_mer${mer}_subset500 data/lang exp/mer$mer/mono_ali exp/mer$mer/tri1 

  #tri1 decoding
  utils/mkgraph.sh data/lang_test exp/mer$mer/tri1 exp/mer$mer/tri1/graph

  for dev in dev_overlap dev_non_overlap; do
    steps/decode.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
      exp/mer$mer/tri1/graph data/$dev exp/mer$mer/tri1/decode_$dev &
  done
fi

if [ $stage -le 11 ]; then
  #tri1 alignment
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri1 exp/mer$mer/tri1_ali 

  #tri2 [a larger model than tri1]
  steps/train_deltas.sh --cmd "$train_cmd" \
    3000 40000 data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri1_ali exp/mer$mer/tri2

  #tri2 decoding
  utils/mkgraph.sh data/lang_test exp/mer$mer/tri2 exp/mer$mer/tri2/graph

  for dev in dev_overlap dev_non_overlap; do
   steps/decode.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
   exp/mer$mer/tri2/graph data/$dev exp/mer$mer/tri2/decode_$dev &
  done
fi

if [ $stage -le 12 ]; then
  #tri2 alignment
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri2 exp/mer$mer/tri2_ali

  # tri3 training [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri1_ali exp/mer$mer/tri3

  #tri3 decoding
  utils/mkgraph.sh data/lang_test exp/mer$mer/tri3 exp/mer$mer/tri3/graph

  for dev in dev_overlap dev_non_overlap; do
   steps/decode.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
   exp/mer$mer/tri3/graph data/$dev exp/mer$mer/tri3/decode_$dev & 
  done
fi

if [ $stage -le 13 ]; then
  #tri3 alignment
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri3 exp/mer$mer/tri3_ali

  #now we start building model with speaker adaptation SAT [fmllr]
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 100000 data/train_mer${mer}_subset500 data/lang exp/mer$mer/tri3_ali exp/mer$mer/tri4

  #sat decoding
  utils/mkgraph.sh data/lang_test exp/mer$mer/tri4 exp/mer$mer/tri4/graph

  for dev in dev_overlap dev_non_overlap; do
    steps/decode_fmllr.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
      exp/mer$mer/tri4/graph data/$dev exp/mer$mer/tri4/decode_$dev &
  done
fi

if [ $stage -le 14 ]; then
  #sat alignment
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/train_mer$mer data/lang exp/mer$mer/tri4 exp/mer$mer/tri4_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    10000 150000 data/train_mer$mer data/lang \
    exp/mer$mer/tri4_ali \
    exp/mer$mer/tri5

  utils/mkgraph.sh data/lang_test exp/mer$mer/tri5{,/graph}

  for dev in dev_overlap dev_non_overlap; do
    steps/decode_fmllr.sh --nj $nDecodeJobs --cmd "$decode_cmd" --config conf/decode.config \
      exp/mer$mer/tri5/graph data/$dev exp/mer$mer/tri5/decode_$dev
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" --config conf/decode.config \
      data/lang_test data/lang_test_fg data/$dev \
      exp/mer$mer/tri5/decode_${dev}{,_fg}
  done
fi

exit 0 

# nnet1 dnn                                                                                                                                
local/nnet/run_dnn.sh $mer


time=$(date +"%Y-%m-%d-%H-%M-%S")
results=baseline.$time
#SCORING IS DONE USING SCLITE
for x in exp/*/*/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done | sort -n -k2 > tmp$$

echo "non_overlap_speech_WER:" > $results
grep decode_dev_non_overlap tmp$$ >> $results
echo "" >> $results
echo "" >> $results
echo "overlap_speech_WER:" >> $results
grep decode_dev_overlap tmp$$ >> $results
echo "" >> $results
rm -fr tmp$$

