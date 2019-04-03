#!/bin/bash

# Copyright 2018    Hossein Hadian
#                   Ashish Arora
#                   Jonathan Chang
# Apache 2.0

set -e
stage=0
nj=30

language_main=Korean
slam_dir=/export/corpora5/slam/SLAM/
yomdle_dir=/export/corpora5/slam/YOMDLE/
corpus_dir=/export/corpora5/handwriting_ocr/corpus_data/ko/
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

./local/check_tools.sh
# Start from stage=-2 for data preparation. This stage stores line images,
# csv files and splits{train,test,train_unsup} data/download/truth_line_image,
# data/download/truth_csv and data/local/splits respectively.
if [ $stage -le -2 ]; then
  echo "$(date): preparing data, obtaining line images and csv files..."
  local/yomdle/create_download_dir.sh --language_main $language_main \
    --slam_dir $slam_dir --yomdle_dir $yomdle_dir
fi

if [ $stage -le -1 ]; then
  echo "$(date): getting corpus text for language modelling..."
  mkdir -p data/local/text/cleaned
  cat $corpus_dir/* > data/local/text/ko.txt
  head -20000 data/local/text/ko.txt > data/local/text/cleaned/val.txt
  tail -n +20000 data/local/text/ko.txt > data/local/text/cleaned/corpus.txt
fi

mkdir -p data/{train,test}/data
if [ $stage -le 0 ]; then
  echo "$0 stage 0: Processing train and test data.$(date)"
  echo " creating text, images.scp, utt2spk and spk2utt"
  #local/prepare_data.sh data/download/
  for set in train test; do
    local/process_data.py data/download/ \
      data/local/splits/${set}.txt data/${set}
    image/fix_data_dir.sh data/${set}
  done
fi

if [ $stage -le 1 ]; then
  echo "$(date) stage 1: getting allowed image widths for e2e training..."
  image/get_image2num_frames.py --feat-dim 40 data/train
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  for set in train test; do
    echo "$(date) Extracting features, creating feats.scp file"
    local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/${set}
    steps/compute_cmvn_stats.sh data/${set} || exit 1;
  done
  image/fix_data_dir.sh data/train
fi

if [ $stage -le 3 ]; then
  echo "$(date) stage 3: BPE preparation"
  # getting non-silence phones.
  cut -d' ' -f2- data/train/text | \
python3 <(
cat << "END"
import os, sys, io;
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8');
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8');
phone_dict = dict();
for line in infile:
    line_vect = line.strip().split();
    for word in line_vect:
        for phone in word:
            phone_dict[phone] = phone;

for phone in phone_dict.keys():
      output.write(phone+ '\n');
END
   ) > data/local/text/cleaned/phones.txt

  cut -d' ' -f2- data/train/text > data/local/text/cleaned/train.txt

  echo "learning BPE..."
  # it is currently learned with only training text but we can also use all corpus text
  # to learn BPE. phones are added so that one isolated occurance of every phone exists.
  cat data/local/text/cleaned/phones.txt data/local/text/cleaned/train.txt | \
    utils/lang/bpe/prepend_words.py | utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$(date) stage 4: applying BPE..."
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

if [ $stage -le 5 ]; then
  echo "$(date) stage 5: Preparing dictionary and lang..."
  local/prepare_dict.sh --dir data/local/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 4 --sil-prob 0.0 --position-dependent-phones false \
    data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi

if [ $stage -le 6 ]; then
  echo "$(date) stage 6: Calling the flat-start chain recipe..."
  local/chain/run_e2e_cnn.sh
fi

if [ $stage -le 7 ]; then
  echo "$(date) stage 7: Aligning the training data using the e2e chain model..."
  steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
    --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
    data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

chunk_width='340,300,200,100'
lang_decode=data/lang
lang_rescore=data/lang_rescore_6g
if [ $stage -le 8 ]; then
  echo "$(date) stage 8: Building a tree and training a regular chain model using the e2e alignments..."
  local/chain/run_cnn_e2eali.sh --chunk_width $chunk_width
fi

if [ $stage -le 9 ]; then
  echo "$(date) stage 9: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/6gram_small.arpa.gz \
                     data/local/dict/lexicon.txt data/lang
  utils/build_const_arpa_lm.sh data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
                               data/lang data/lang_rescore_6g
fi

if [ $stage -le 10 ] && $decode_e2e; then
  echo "$(date) stage 10: decoding end2end setup..."

  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_decode \
    exp/chain/e2e_cnn_1a/ exp/chain/e2e_cnn_1a/graph || exit 1;

  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj 30 --cmd "$cmd" --beam 12 \
    exp/chain/e2e_cnn_1a/graph data/test exp/chain/e2e_cnn_1a/decode_test || exit 1;

  steps/lmrescore_const_arpa.sh --cmd "$cmd" $lang_decode $lang_rescore \
                                data/test exp/chain/e2e_cnn_1a/decode_test{,_rescored} || exit 1

  echo "Done. Date: $(date). Results:"
  local/chain/compare_wer.sh exp/chain/e2e_cnn_1a/
fi

if [ $stage -le 11 ] && $decode_chain; then
  echo "$(date) stage 11: decoding chain alignment setup..."

  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_decode \
    exp/chain/cnn_e2eali_1a/ exp/chain/cnn_e2eali_1a/graph || exit 1;

  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj 30 --cmd "$cmd" --beam 12 \
    exp/chain/cnn_e2eali_1a/graph data/test exp/chain/cnn_e2eali_1a/decode_test || exit 1;

  steps/lmrescore_const_arpa.sh --cmd "$cmd" $lang_decode $lang_rescore \
                                data/test exp/chain/cnn_e2eali_1a/decode_test{,_rescored} || exit 1

  echo "Done. Date: $(date). Results:"
  local/chain/compare_wer.sh exp/chain/cnn_e2eali_1a
fi
