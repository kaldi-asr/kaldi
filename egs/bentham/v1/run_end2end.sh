#!/usr/bin/env bash
# Copyright     2018    Ashish Arora (Johns Hopkins University)
#               2018    Desh Raj (Johns Hopkins University)

set -e
stage=0
nj=20
# bentham_hwr_database points to the official database path on the JHU grid. If you have not
# already downloaded the data, you will have to first download it and then name the Images
# and Ground Truth zipped files as images.zip and gt.zip. Then, point the path below to the
# location where your zipped files are present on the grid.
bentham_hwr_database=/export/corpora5/handwriting_ocr/hwr1/ICDAR-HTR-Competition-2015
# bentham_text_database points to the database path on the JHU grid.
# It contains all of the written works of Bentham, and can be used to train
# an LM for the HWR task. We have provided a script which downloads the data
# and saves it to the location provided below.
bentham_text_corpus=/export/corpora5/handwriting_ocr/hwr1/ICDAR-HTR-Competition-2015/Bentham-Text

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
. ./utils/parse_options.sh  # e.g. this parses the above options
                            # if supplied.


./local/check_tools.sh

if [ $stage -le 0 ]; then
  echo "$0: Preparing data..."
  local/prepare_data.sh --database-dir $bentham_hwr_database \
    --text-corpus-dir $bentham_text_corpus
fi

if [ $stage -le 1 ]; then
  image/get_image2num_frames.py data/train  # This will be needed for the next command
  # The next command creates a "allowed_lengths.txt" file in data/train
  # which will be used by local/make_features.py to enforce the images to
  # have allowed lengths. The allowed lengths will be spaced by 10% difference in length.
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train
  echo "$(date) Extracting features, creating feats.scp file"
  for dataset in train val test; do
    local/extract_features.sh --nj $nj --cmd "$cmd" --feat-dim 40 data/$dataset  
    steps/compute_cmvn_stats.sh data/$dataset
  done
  utils/fix_data_dir.sh data/train
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
    utils/lang/bpe/prepend_words.py | \
    utils/lang/bpe/learn_bpe.py -s 700 > data/local/bpe.txt
  for set in test train val; do
    cut -d' ' -f1 data/$set/text > data/$set/ids
    cut -d' ' -f2- data/$set/text | \
      utils/lang/bpe/prepend_words.py | utils/lang/bpe/apply_bpe.py -c data/local/bpe.txt \
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
  # This recipe uses byte-pair encoding, the silences are part of the words' pronunciations.
  # So we set --sil-prob to 0.0
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
    data/local/dict "<sil>" data/lang/temp data/lang
  silphonelist=`cat data/lang/phones/silence.csl`
  nonsilphonelist=`cat data/lang/phones/nonsilence.csl`
  local/gen_topo.py 8 4 4 $nonsilphonelist $silphonelist data/lang/phones.txt >data/lang/topo
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang

  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/6gram_big.arpa.gz \
    data/local/dict/lexicon.txt data/lang
  utils/build_const_arpa_lm.sh data/local/local_lm/data/arpa/6gram_unpruned.arpa.gz \
    data/lang data/lang_rescore_6g
fi

if [ $stage -le 5 ]; then
  echo "$0: Calling the flat-start chain recipe..."
  local/chain/run_e2e_cnn.sh
fi

if [ $stage -le 6 ]; then
  echo "$0: Aligning the training data using the e2e chain model..."
  steps/nnet3/align.sh --nj 50 --cmd "$cmd" \
                       --use-gpu false \
                       --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
                       data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

if [ $stage -le 7 ]; then
  echo "$0: Building a tree and training a regular chain model using the e2e alignments..."
  local/chain/run_cnn_e2eali.sh
fi
