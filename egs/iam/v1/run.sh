#!/bin/bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian

set -e
stage=0
nj=20
decode_gmm=false
username=
password=
# iam_database points to the database path on the JHU grid. If you have not
# already downloaded the database you can set it to a local directory
# like "data/download" and follow the instructions
# in "local/prepare_data.sh" to download the database:
iam_database=/export/corpora5/handwriting_ocr/IAM
# wellington_database points to the database path on the JHU grid. The Wellington
# corpus contains two directories WWC and WSC (Wellington Written and Spoken Corpus).
# This corpus is of written NZ English that can be purchased here:
# "https://www.victoria.ac.nz/lals/resources/corpora-default"
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
  echo "$0: Preparing the test and train feature files..."
  for dataset in train test; do
    local/make_features.py data/$dataset --feat-dim 40 | \
      copy-feats --compress=true --compression-method=7 \
                 ark:- ark,scp:data/$dataset/data/images.ark,data/$dataset/feats.scp
    steps/compute_cmvn_stats.sh data/$dataset
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Estimating a language model for decoding..."
  # We do this stage before dict preparation because prepare_dict.sh
  # generates the lexicon from pocolm's wordlist
  local/train_lm.sh --vocab-size 50k
fi

if [ $stage -le 3 ]; then
  echo "$0: Preparing dictionary and lang..."

  # This is for training. Use a large vocab size, e.g. 500k to include all the
  # training words:
  local/prepare_dict.sh --vocab-size 500k --dir data/local/dict  # this is for training
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
                        data/local/dict "<unk>" data/lang/temp data/lang

  # This is for decoding. We use a 50k lexicon to be consistent with the papers
  # reporting WERs on IAM:
  local/prepare_dict.sh --vocab-size 50k --dir data/local/dict_50k  # this is for decoding
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
                        data/local/dict_50k "<unk>" data/lang_test/temp data/lang_test
  utils/format_lm.sh data/lang_test data/local/local_lm/data/arpa/3gram_big.arpa.gz \
                     data/local/dict_50k/lexicon.txt data/lang_test

  echo "$0: Preparing the unk model for open-vocab decoding..."
  utils/lang/make_unk_lm.sh --ngram-order 4 --num-extra-ngrams 7500 \
                            data/local/dict_50k exp/unk_lang_model
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 \
                        --unk-fst exp/unk_lang_model/unk_fst.txt \
                        data/local/dict_50k "<unk>" data/lang_unk/temp data/lang_unk
  cp data/lang_test/G.fst data/lang_unk/G.fst
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd --totgauss 10000 data/train \
    data/lang exp/mono
fi

if [ $stage -le 5 ] && $decode_gmm; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/mono/graph data/test \
    exp/mono/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd $cmd 500 20000 data/train data/lang \
    exp/mono_ali exp/tri
fi

if [ $stage -le 7 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri exp/tri/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri/graph data/test \
    exp/tri/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd data/train data/lang \
    exp/tri exp/tri_ali

  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/tri_ali exp/tri2
fi

if [ $stage -le 9 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph

  steps/decode.sh --nj $nj --cmd $cmd exp/tri2/graph \
    data/test exp/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd $cmd 500 20000 \
    data/train data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 11 ] && $decode_gmm; then
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph

  steps/decode_fmllr.sh --nj $nj --cmd $cmd exp/tri3/graph \
    data/test exp/tri3/decode_test
fi

if [ $stage -le 12 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

if [ $stage -le 13 ]; then
  local/chain/run_cnn_1a.sh --lang-test lang_unk
fi

if [ $stage -le 14 ]; then
  local/chain/run_cnn_chainali_1c.sh --chain-model-dir exp/chain/cnn_1a --stage 2
fi
