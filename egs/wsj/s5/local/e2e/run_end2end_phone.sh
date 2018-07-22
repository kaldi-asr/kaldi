#!/bin/bash
# Copyright 2017    Hossein Hadian

# This top-level script demonstrates end-to-end LF-MMI training (specifically
# single-stage flat-start LF-MMI models) on WSJ. It is basically like
# "../run.sh" except it does not train any GMM or SGMM models and after
# doing data/dict preparation and feature extraction goes straight to
# flat-start chain training.
# It uses a phoneme-based lexicon just like "../run.sh" does.

set -euo pipefail


stage=0
trainset=train_si284
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

#wsj0=/ais/gobi2/speech/WSJ/csr_?_senn_d?
#wsj1=/ais/gobi2/speech/WSJ/csr_senn_d?

#wsj0=/mnt/matylda2/data/WSJ0
#wsj1=/mnt/matylda2/data/WSJ1

#wsj0=/data/corpora0/LDC93S6B
#wsj1=/data/corpora0/LDC94S13B

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

. ./path.sh
. utils/parse_options.sh


# This is just like stage 0 in run.sh except we do mfcc extraction later
# We use the same suffixes as in run.sh (i.e. _nosp) for consistency

if [ $stage -le 0 ]; then
  # data preparation.
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
  local/wsj_prepare_dict.sh --dict-suffix "_nosp"
  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp
  local/wsj_format_data.sh --lang-suffix "_nosp"
  echo "Done formatting the data."

  local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1
  utils/prepare_lang.sh data/local/dict_nosp_larger \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger \
                        data/lang_nosp_bd
  local/wsj_train_lms.sh --dict-suffix "_nosp"
  local/wsj_format_local_lms.sh --lang-suffix "_nosp"
  echo "Done exteding the dictionary and formatting LMs."
fi

if [ $stage -le 1 ]; then
  # make MFCC features for the test data. Only hires since it's flat-start.
  echo "$0: extracting MFCC features for the test sets"
  for x in test_eval92 test_eval93 test_dev93; do
    mv data/$x data/${x}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
                       --mfcc-config conf/mfcc_hires.conf data/${x}_hires
    steps/compute_cmvn_stats.sh data/${x}_hires
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command

  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
                                                 data/${trainset}_spe2e_hires
  cat data/${trainset}_spe2e_hires/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spe2e_hires
fi

if [ $stage -le 3 ]; then
  echo "$0: extracting MFCC features for the training data"
  steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
                     --cmd "$train_cmd" data/${trainset}_spe2e_hires
  steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires
fi

if [ $stage -le 4 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh
fi
