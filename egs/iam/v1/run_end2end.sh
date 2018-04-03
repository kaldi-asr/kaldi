#!/bin/bash
# Copyright 2017    Hossein Hadian

set -e
stage=0
nj=20
username=
password=
# iam_database points to the database path on the JHU grid. If you have not
# already downloaded the database you can set it to a local directory
# like "data/download" and follow the instructions
# in "local/prepare_data.sh" to download the database:
iam_database=/export/corpora5/handwriting_ocr/IAM

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.


./local/check_tools.sh

if [ $stage -le 0 ]; then
  echo "$0: Preparing data..."
  local/prepare_data.sh --download-dir "$iam_database" \
    --username "$username" --password "$password"
fi
mkdir -p data/{train,test}/data

if [ $stage -le 1 ]; then
  get_image2num_frames.py data/train  # This will be needed for the next command
  # The next command creates a "allowed_lengths.txt" file in data/train
  # which will be used by local/make_features.py to enforce the images to
  # have allowed lengths. The allowed lengths will be spaced by 10% difference in length.
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  echo "$0: Preparing the test and train feature files..."
  for dataset in train test; do
    local/make_features.py data/$dataset --feat-dim 40 | \
      copy-feats --compress=true --compression-method=7 \
                 ark:- ark,scp:data/$dataset/data/images.ark,data/$dataset/feats.scp
    steps/compute_cmvn_stats.sh data/$dataset
  done
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --sil-prob 0.95 \
                        data/local/dict "<unk>" data/lang/temp data/lang
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding..."
  local/train_lm.sh
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_big.arpa.gz \
                     data/local/dict/lexicon.txt data/lang_test
fi


if [ $stage -le 4 ]; then
  echo "$0: estimating phone language model for the denominator graph"
  mkdir -p exp/chain/e2e_base/log
  $cmd exp/chain/e2e_base/log/make_phone_lm.log \
  cat data/train/text \| \
    steps/nnet3/chain/e2e/text_to_phones.py data/lang \| \
    utils/sym2int.pl -f 2- data/lang/phones.txt \| \
    chain-est-phone-lm --num-extra-lm-states=1000 \
                       ark:- exp/chain/e2e_base/phone_lm.fst
fi

if [ $stage -le 5 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/run_flatstart_cnn1a.sh
fi
