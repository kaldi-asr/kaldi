#!/bin/sh
# Copyright 2017    Hossein Hadian

set -e


stage=0
trainset=train_si284
. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./utils/parse_options.sh  # e.g. this parses the --stage option if supplied.



#wsj0=/ais/gobi2/speech/WSJ/csr_?_senn_d?
#wsj1=/ais/gobi2/speech/WSJ/csr_senn_d?

#wsj0=/mnt/matylda2/data/WSJ0
#wsj1=/mnt/matylda2/data/WSJ1

#wsj0=/data/corpora0/LDC93S6B
#wsj1=/data/corpora0/LDC94S13B

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# This is just like stage 0 in run.sh except we skip mfcc extraction for training data

if [ $stage -le 0 ]; then
  # data preparation.
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;
  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;
  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  (
    local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1  && \
      utils/prepare_lang.sh data/local/dict_nosp_larger \
                            "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
      local/wsj_train_lms.sh --dict-suffix "_nosp" &&
      local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  ) &

  # Now make MFCC features. Only hires since it's end to end
  echo "$0: extracting MFCC features for the test sets"
  for x in test_eval92 test_eval93 test_dev93; do
    mv data/$x data/${x}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
                       --mfcc-config conf/mfcc_hires.conf data/${x}_hires
    steps/compute_cmvn_stats.sh data/${x}_hires
  done
  wait
fi

if [ $stage -le 1 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  mkdir -p exp/chain/e2e_base
  utils/data/get_utt2dur.sh data/$trainset  # necessary for next command

  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length
  python utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
         data/${trainset}_spEx_hires
  cat data/${trainset}_spEx_hires/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${trainset}_spEx_hires/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spEx_hires
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting MFCC features for the training data"
  steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
                     --cmd "$train_cmd" data/${trainset}_spEx_hires
  steps/compute_cmvn_stats.sh data/${trainset}_spEx_hires
fi

if [ $stage -le 3 ]; then
  echo "$0: estimating phone language model for the denominator graph"
  cat data/$trainset/text | \
    utils/text_to_phones.py data/lang_nosp data/local/dict_nosp/lexicon.txt | \
    utils/sym2int.pl -f 2- data/lang_nosp/phones.txt | \
    chain-est-phone-lm --num-extra-lm-states=2000 \
                       ark:- exp/chain/e2e_base/phone_lm.fst
fi
