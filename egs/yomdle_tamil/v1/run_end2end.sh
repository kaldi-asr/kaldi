#!/bin/bash

# Copyright 2018    Hossein Hadian
#                   Ashish Arora
#                   Jonathan Chang
# Apache 2.0

set -e
stage=0
nj=80

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
mkdir -p data/{train,train_unsup,test}/data
mkdir -p data/local/backup

if [ $stage -le -1 ]; then
  echo "$0: creating line images for shared model and unsupervised training...$(date)"
  local/create_download_dir.sh --language_main Tamil
  echo "$0: getting text for language modelling...$(date)"
  cat /export/corpora5/handwriting_ocr/corpus_data/ta/* > data/local/text/ta.txt
  head -20000 data/local/text/ta.txt > data/local/text/val.txt
  tail -n +20000 data/local/text/ta.txt > data/local/text/corpus.txt
fi

if [ $stage -le 0 ]; then
  echo "stage 0: Processing train, train unsupervised and test data...$(date)"
  local/prepare_data.sh --language tamil
fi

if [ $stage -le 1 ]; then
  image/get_image2num_frames.py --feat-dim 40 data/train
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  for set in train train_unsup; do
    echo "$0: Extracting features and calling compute_cmvn_stats for dataset:  $set. "
    echo "Date: $(date)."
    local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/${set}
    steps/compute_cmvn_stats.sh data/${set} || exit 1;
    #image/ocr/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    #image/ocr/make_features.py data/$set/images.scp --feat-dim 40 \
    #  --allowed_len_file_path data/$set/allowed_lengths.txt --no-augment | \
    #  copy-feats --compress=true --compression-method=7 \
    #    ark:- ark,scp:data/$set/data/images.ark,data/$set/feats.scp
    #steps/compute_cmvn_stats.sh data/$set || exit 1;
  done
  utils/fix_data_dir.sh data/train

  local/make_features.py data/test/images.scp --feat-dim 40 \
      --allowed_len_file_path data/test/allowed_lengths.txt  --no-augment | \
      copy-feats --compress=true --compression-method=7 \
               ark:- ark,scp:data/test/data/images.ark,data/test/feats.scp
fi

if [ $stage -le 2 ]; then
  echo "stage 2: BPE preparation  $(date)"
  cp -r data/train data/local/backup1/
  cp -r data/test data/local/backup1/
  cp -r data/train_unsup data/local/backup1/

  cut -d' ' -f2- data/train/text | \
    local/get_phones.py > data/local/text/cleaned/phones.txt
  cut -d' ' -f2- data/train/text > data/local/text/cleaned/train.txt

  echo ": Processing corpus text"
  cat data/local/text/corpus.txt | \
    local/process_corpus.py > data/local/text/cleaned/corpus.txt
  cat data/local/text/val.txt | \
    local/process_corpus.py > data/local/text/cleaned/val.txt

  echo ": learning BPE"
  cat data/local/text/cleaned/phones.txt data/local/text/cleaned/train.txt | \
    local/prepend_words.py | utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "stage 3: applying BPE...$(date)"
  echo "applying BPE on train test text..."
  for set in test train; do
      cut -d' ' -f1 data/$set/text > data/$set/ids
      cut -d' ' -f2- data/$set/text | local/prepend_words.py | \
        utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
        sed 's/@@//g' > data/$set/bpe_text
      mv data/$set/text data/$set/text.old
      paste -d' ' data/$set/ids data/$set/bpe_text > data/$set/text
  done

  echo "applying BPE to corpus text"
  cat data/local/text/cleaned/corpus.txt | local/prepend_words.py | \
    utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
    sed 's/@@//g' > data/local/text/cleaned/bpe_corpus.txt
  cat data/local/text/cleaned/val.txt | local/prepend_words.py | \
    utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt | \
    sed 's/@@//g' > data/local/text/cleaned/bpe_val.txt
fi

if [ $stage -le 4 ]; then
  echo "stage 4: Preparing dictionary and lang... $(date)"
  local/prepare_dict.sh --dir data/local/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
      data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi

if [ $stage -le 5 ]; then
  echo "stage 5: Estimating a language model for decoding...$(date)"
  local/train_lm.sh --dir data/local/local_lm --order 3
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
      data/local/dict/lexicon.txt data/lang_test

  #local/train_lm.sh --dir data/local/local_lm --order 6
  #utils/build_const_arpa_lm.sh data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
  #                             data/lang data/lang_rescore_6g
fi

if [ $stage -le 6 ]; then
  echo "stage 6: Calling the flat-start chain recipe...$(date)"
  local/chain/run_e2e_cnn.sh
fi

if [ $stage -le 7 ]; then
  echo "stage 7: Aligning the training data using the e2e chain model... $(date)"
  steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
      --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
      data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

if [ $stage -le 8 ]; then
  echo "stage 8: Building a tree and training a regular chain model using the e2e alignments...$(date)"
  local/chain/run_cnn_e2eali_1b.sh
fi
