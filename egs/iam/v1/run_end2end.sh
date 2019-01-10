#!/bin/bash
# Copyright 2017    Hossein Hadian

set -e
stage=0
nj=20
username=
password=
process_aachen_split=false
overwrite=false
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
train_set=train_aug
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.
./local/check_tools.sh
if [ $stage -le 0 ]; then

 if [ -f data/train/text ] && ! $overwrite; then
    echo "$0: Not processing, probably script have run from wrong stage"
    echo "Exiting with status 1 to avoid data corruption"
    exit 1;
  fi

  echo "$0: Preparing data..."
  local/prepare_data.sh --download-dir "$iam_database" \
    --wellington-dir "$wellington_database" \
    --username "$username" --password "$password" \
    --process_aachen_split $process_aachen_split
fi
mkdir -p data/{train,test,val}/data

if [ $stage -le 1 ]; then
  echo "$0: $(date) stage 1: getting allowed image widths for e2e training..."
  image/get_image2num_frames.py --feat-dim 40 data/train # This will be needed for the next command
  # The next command creates a "allowed_lengths.txt" file in data/train
  # which will be used by local/make_features.py to enforce the images to
  # have allowed lengths. The allowed lengths will be spaced by 10% difference in length.
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  echo "$0: $(date) Extracting features, creating feats.scp file"
  local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/train
  steps/compute_cmvn_stats.sh data/train || exit 1;
  for set in val test; do
    local/extract_features.sh --nj $nj --cmd "$cmd" --augment true \
    --feat-dim 40 data/${set}
    steps/compute_cmvn_stats.sh data/${set} || exit 1;
  done
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  for set in train; do
    echo "$0: $(date) stage 2: Performing augmentation, it will double training data"
    local/augment_data.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/${set} data/${set}_aug data
    steps/compute_cmvn_stats.sh data/${set}_aug || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding..."
  # We do this stage before dict preparation because prepare_dict.sh
  # generates the lexicon from pocolm's wordlist
  local/train_lm.sh --vocab-size 50k
fi

if [ $stage -le 4 ]; then
  echo "$0: Preparing dictionary and lang..."
  # This is for training. Use a large vocab size, e.g. 500k to include all the
  # training words:
  local/prepare_dict.sh --vocab-size 500k --dir data/local/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
                        data/local/dict "<unk>" data/lang/temp data/lang
  silphonelist=`cat data/lang/phones/silence.csl`
  nonsilphonelist=`cat data/lang/phones/nonsilence.csl`
  local/gen_topo.py 8 4 4 $nonsilphonelist $silphonelist data/lang/phones.txt >data/lang/topo

  # This is for decoding. We use a 50k lexicon to be consistent with the papers
  # reporting WERs on IAM.
  local/prepare_dict.sh --vocab-size 50k --dir data/local/dict_50k
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.95 \
                        data/local/dict_50k "<unk>" data/lang_test/temp data/lang_test
  utils/format_lm.sh data/lang_test data/local/local_lm/data/arpa/3gram_big.arpa.gz \
                     data/local/dict_50k/lexicon.txt data/lang_test

  echo "$0: Preparing the unk model for open-vocab decoding..."
  utils/lang/make_unk_lm.sh --ngram-order 4 --num-extra-ngrams 7500 \
                            data/local/dict_50k exp/unk_lang_model
  utils/prepare_lang.sh --unk-fst exp/unk_lang_model/unk_fst.txt \
                        data/local/dict_50k "<unk>" data/lang_unk/temp data/lang_unk

  silphonelist=`cat data/lang/phones/silence.csl`
  nonsilphonelist=`cat data/lang/phones/nonsilence.csl`
  local/gen_topo.py 8 4 4 $nonsilphonelist $silphonelist data/lang_unk/phones.txt >data/lang_unk/topo
  cp data/lang_test/G.fst data/lang_unk/G.fst
fi

if [ $stage -le 5 ]; then
  echo "$0: Calling the flat-start chain recipe..."
  local/chain/run_e2e_cnn.sh --train_set $train_set
fi

if [ $stage -le 6 ]; then
  echo "$0: Aligning the training data using the e2e chain model..."
  steps/nnet3/align.sh --nj 50 --cmd "$cmd" \
                       --use-gpu false \
                       --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
                       data/$train_set data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

if [ $stage -le 7 ]; then
  echo "$0: Building a tree and training a regular chain model using the e2e alignments..."
  local/chain/run_cnn_e2eali.sh --train_set $train_set
fi
