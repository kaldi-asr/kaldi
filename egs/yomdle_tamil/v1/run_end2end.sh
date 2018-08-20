#!/bin/bash

# Copyright 2018    Hossein Hadian
#                   Ashish Arora
#                   Jonathan Chang
# Apache 2.0

set -e
stage=0
nj=30

language_main=Tamil
slam_dir=/export/corpora5/slam/SLAM/
yomdle_dir=/export/corpora5/slam/YOMDLE/
corpus_dir=/export/corpora5/handwriting_ocr/corpus_data/ta/

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

./local/check_tools.sh
mkdir -p data/{train,test}/data

# Start from stage=-2 for extracting line images from page image and 
# creating csv files for transcriptions.
if [ $stage -le -2 ]; then
  echo "$(date): extracting line images for shared model and semi-supervised training..."
  local/yomdle/create_download_dir.sh --language_main $language_main \
    --slam_dir $slam_dir --yomdle_dir $yomdle_dir
fi

if [ $stage -le -1 ]; then
  echo "$(date): getting corpus text for language modelling..."
  mkdir -p data/local/text/cleaned
  cat $corpus_dir/* > data/local/text/ta.txt
  head -20000 data/local/text/ta.txt > data/local/text/val.txt
  tail -n +20000 data/local/text/ta.txt > data/local/text/corpus.txt
fi

if [ $stage -le 0 ]; then
  echo "$(date) stage 0: Processing train and test data."
  echo " creating text, images.scp, utt2spk and spk2utt"
  # removing empty transcription line images from train and test set.
  # It can cause error while applying BPE.
  for set in train test; do
    local/process_data.py data/download/ \
      data/local/splits/${set}.txt data/${set}
    image/fix_data_dir.sh data/${set}
  done
fi

if [ $stage -le 1 ]; then
  echo "$(date) stage 1: Obtaining image groups. calling get_image2num_frames..."
  image/get_image2num_frames.py --feat-dim 40 data/train
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  for set in train test; do
    echo "$(date) Extracting features and calling compute_cmvn_stats"
    local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/${set}
    steps/compute_cmvn_stats.sh data/${set} || exit 1;
  done
  image/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  echo "$(date) stage 2: BPE preparation"
  # getting non-silence phones.
  cat data/train/text | \
  perl -ne '@A = split; shift @A; for(@A) {print join("\n", split(//)), "\n";}' | \
  sort -u > data/local/text/cleaned/phones.txt

  cut -d' ' -f2- data/train/text > data/local/text/cleaned/train.txt

  echo "Processing corpus text..."
  # we are removing the lines from the corpus which which have
  # phones other than the phones in data/local/text/cleaned/phones.txt.
  cat data/local/text/corpus.txt | \
    local/process_corpus.py > data/local/text/cleaned/corpus.txt
  cat data/local/text/val.txt | \
    local/process_corpus.py > data/local/text/cleaned/val.txt

  echo "learning BPE..."
  # it is currently learned with only training text but we can also use all corpus text
  # to learn BPE. phones are added so that one isolated occurance of every phone exists.
  cat data/local/text/cleaned/phones.txt data/local/text/cleaned/train.txt | \
    utils/lang/bpe/prepend_words.py | utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$(date) stage 3: applying BPE..."
  echo "applying BPE on train, test text..."
  for set in test train; do
    cut -d' ' -f1 data/$set/text > data/$set/ids
    cut -d' ' -f2- data/$set/text | utils/lang/bpe/prepend_words.py | \
      utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
      sed 's/@@//g' > data/$set/bpe_text
    mv data/$set/text data/$set/text.old
    paste -d' ' data/$set/ids data/$set/bpe_text > data/$set/text
    rm -f data/$set/bpe_text data/$set/ids
  done

  echo "applying BPE to corpus text..."
  cat data/local/text/cleaned/corpus.txt | utils/lang/bpe/prepend_words.py | \
    utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
    sed 's/@@//g' > data/local/text/cleaned/bpe_corpus.txt
  cat data/local/text/cleaned/val.txt | utils/lang/bpe/prepend_words.py | \
    utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
    sed 's/@@//g' > data/local/text/cleaned/bpe_val.txt
fi

if [ $stage -le 4 ]; then
  echo "$(date) stage 4: Preparing dictionary and lang..."
  local/prepare_dict.sh --dir data/local/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
    data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi

if [ $stage -le 5 ]; then
  echo "$(date) stage 5: Estimating a language model for decoding..."
  local/train_lm.sh --dir data/local/local_lm --order 3 \
    bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.016,0.938,0.779,0.027,0.001,0.000,0.930,0.647,0.308,0.101"
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test

  local/train_lm.sh --dir data/local/local_lm_6g --order 6 \
    bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.031,0.860,0.678,0.194,0.037,0.006,0.928,0.712,0.454,0.220,0.926,0.844,0.749,0.358,0.966,0.879,0.783,0.544,0.966,0.826,0.674,0.450"
  utils/build_const_arpa_lm.sh data/local/local_lm_6g/data/arpa/6gram_big.arpa.gz \
                               data/lang data/lang_rescore_6g
fi

if [ $stage -le 6 ]; then
  echo "$(date) stage 6: Calling the flat-start chain recipe..."
  local/chain/run_e2e_cnn.sh
fi

if [ $stage -le 7 ]; then
  echo "$(date) stage 7: Aligning the training data using the e2e chain model..."
  steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
    --use-gpu false \
    --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
    data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

if [ $stage -le 8 ]; then
  echo "$(date) stage 8: Building a tree and training a regular chain model using the e2e alignments..."
  local/chain/run_cnn_e2eali.sh
fi
