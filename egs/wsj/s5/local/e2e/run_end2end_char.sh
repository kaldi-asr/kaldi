#!/bin/bash
# Copyright 2017    Hossein Hadian

# This top-level script demonstrates character-based end-to-end LF-MMI training
# (specifically single-stage flat-start LF-MMI models) on WSJ. It is exactly
# like "run_end2end_phone.sh" excpet it uses a trivial grapheme-based
# (i.e. character-based) lexicon and a stronger neural net (i.e. TDNN-LSTM)

set -euo pipefail


stage=0
trainset=train_si284
. ./cmd.sh ## You'll want to change cmd.sh to something that will work
           ## on your system. This relates to the queue.

#wsj0=/ais/gobi2/speech/WSJ/csr_?_senn_d?
#wsj1=/ais/gobi2/speech/WSJ/csr_senn_d?

#wsj0=/mnt/matylda2/data/WSJ0
#wsj1=/mnt/matylda2/data/WSJ1

#wsj0=/data/corpora0/LDC93S6B
#wsj1=/data/corpora0/LDC94S13B

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

. ./utils/parse_options.sh
. ./path.sh

# We use the suffix _nosp for the phoneme-based dictionary and
# lang directories (for consistency with run.sh) and the suffix
# _char for character-based dictionary and lang directories.

if [ $stage -le 0 ]; then
  [[ -d data/local/data ]] || \
    local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
  [[ -f data/local/dict_nosp/lexicon.txt ]] || \
    local/wsj_prepare_dict.sh --dict-suffix "_nosp"

  local/wsj_prepare_char_dict.sh
  utils/prepare_lang.sh data/local/dict_char \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_char data/lang_char
  local/wsj_format_data.sh --lang-suffix "_char"
  echo "$0: Done preparing data & lang."
fi

if [ $stage -le 1 ]; then
  local/wsj_extend_char_dict.sh $wsj1/13-32.1 data/local/dict_char \
                              data/local/dict_char_larger
  utils/prepare_lang.sh data/local/dict_char_larger \
                        "<SPOKEN_NOISE>" data/local/lang_larger_tmp \
                        data/lang_char_bd
  # Note: this will overwrite data/local/local_lm:
  local/wsj_train_lms.sh --dict-suffix "_char"
  local/wsj_format_local_lms.sh --lang-suffix "_char"
  echo "$0: Done extending the vocabulary."
fi

if [ $stage -le 2 ]; then
  # make MFCC features for the test data. Only hires since it's flat-start.
  if [ -f data/test_eval92_hires/feats.scp ]; then
    echo "$0: It seems that features for the test sets already exist."
    echo "skipping this stage..."
  else
    echo "$0: extracting MFCC features for the test sets"
    for x in test_eval92 test_eval93 test_dev93; do
      mv data/$x data/${x}_hires
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
                         --mfcc-config conf/mfcc_hires.conf data/${x}_hires
      steps/compute_cmvn_stats.sh data/${x}_hires
    done
  fi
fi

if [ -f data/${trainset}_spEx_hires/feats.scp ]; then
  echo "$0: It seems that features for the perturbed training data already exist."
  echo "If you want to extract them anyway, remove them first and run this"
  echo "stage again. Skipping this stage..."
else
  if [ $stage -le 3 ]; then
    echo "$0: perturbing the training data to allowed lengths..."
    utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command

    # 12 in the following command means the allowed lengths are spaced
    # by 12% change in length.
    python utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
           data/${trainset}_spe2e_hires
    cat data/${trainset}_spe2e_hires/utt2dur | \
      awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
  fi

  if [ $stage -le 4 ]; then
    echo "$0: extracting MFCC features for the training data..."
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${trainset}_spe2e_hires
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires
  fi
fi

if [ $stage -le 5 ]; then
  echo "$0: estimating character language model for the denominator graph"
  mkdir -p exp/chain/e2e_base/log
  $train_cmd exp/chain/e2e_base/log/make_char_lm.log \
  cat data/$trainset/text \| \
    steps/nnet3/chain/e2e/text_to_phones.py data/lang_char \| \
    utils/sym2int.pl -f 2- data/lang_char/phones.txt \| \
    chain-est-phone-lm --num-extra-lm-states=2000 \
                       ark:- exp/chain/e2e_base/char_lm.fst
fi

if [ $stage -le 6 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_lstm_flatstart.sh
fi
