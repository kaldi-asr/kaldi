#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian

set -e
stage=0
nj=70
decode_gmm=false
username=
password=
iam_database=/export/corpora5/handwriting_ocr/IAM
wellington_database=/export/corpora5/Wellington/WWC/

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.

./local/check_tools.sh

if [ $stage -le 0 ]; then
  echo "$0: Preparing data..."
  local/prepare_data.sh --download-dir "$iam_database" \
    --wellington-dir "$wellington_database" \
    --username "$username" --password "$password"
fi
mkdir -p data/{train,test}/data

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

if [ $stage -le 2 ]; then
  for set in train; do
    echo "$(date) stage 2: Performing augmentation, it will double training data"
    local/augment_data.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/${set} data/${set}_aug data
    steps/compute_cmvn_stats.sh data/${set}_aug || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing BPE..."
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
   ) > data/local/phones.txt

  cut -d' ' -f2- data/train/text > data/local/train_data.txt
  cat data/local/phones.txt data/local/train_data.txt | \
    local/prepend_words.py | \
    utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt
  for set in test train val train_aug; do
    cut -d' ' -f1 data/$set/text > data/$set/ids
    cut -d' ' -f2- data/$set/text | \
      local/prepend_words.py | utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt \
      | sed 's/@@//g' > data/$set/bpe_text
    mv data/$set/text data/$set/text.old
    paste -d' ' data/$set/ids data/$set/bpe_text > data/$set/text
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
fi

if [ $stage -le 4 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
                        data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang

  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/6gram_big.arpa.gz \
                     data/local/dict/lexicon.txt data/lang
  utils/build_const_arpa_lm.sh data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
                               data/lang data/lang_rescore_6g
fi

if [ $stage -le 5 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train_aug \
    data/lang exp/mono
fi

if [ $stage -le 6 ] && $decode_gmm; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train_aug data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd 500 20000 data/train_aug data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 8 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train_aug data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train_aug data/lang exp/tri_ali exp/tri2
fi

if [ $stage -le 10 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri2/graph \
    data/test exp/tri2/decode_test
fi

if [ $stage -le 11 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train_aug data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd $cmd 500 20000 \
    data/train_aug data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 12 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

  steps/decode_fmllr.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 13 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train_aug data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 14 ]; then
  local/chain/run_cnn_1a.sh --train_set train_aug
fi

if [ $stage -le 15 ]; then
  local/chain/run_cnn_chainali_1d.sh --train_set train_aug \
    --chain-model-dir exp/chain/cnn_1a --stage 2
fi
