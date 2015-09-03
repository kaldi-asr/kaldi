#!/bin/bash

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. cmd.sh

mfccdir=mfcc

# Make "per-utterance" versions of the test sets where the speaker
# information corresponds to utterances-- to demonstrate adaptation on
# short utterances, particularly for basis fMLLR
for x in test_eval92 test_eval93 test_dev93 ; do
  y=${x}_utt
  rm -r data/$y
  cp -r data/$x data/$y
  cat data/$x/utt2spk | awk '{print $1, $1;}' > data/$y/utt2spk;
  cp data/$y/utt2spk data/$y/spk2utt;
  steps/compute_cmvn_stats.sh data/$y exp/make_mfcc/$y $mfccdir || exit 1; 
done


 # basis fMLLR experiments.
 # First a baseline: decode per-utterance with normal fMLLR.
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_dev93_utt \
  exp/tri3b/decode${lang_suffix}_tgpr_dev93_utt || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_eval92_utt \
  exp/tri3b/decode${lang_suffix}_tgpr_eval92_utt || exit 1;

 # get the fMLLR basis.
steps/get_fmllr_basis.sh --cmd "$train_cmd" \
  data/train_si84 data/lang${lang_suffix} exp/tri3b

 # decoding tri3b with basis fMLLR
steps/decode_basis_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_dev93 \
  exp/tri3b/decode${lang_suffix}_tgpr_dev93_basis || exit 1;
steps/decode_basis_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_eval92 \
  exp/tri3b/decode${lang_suffix}_tgpr_eval92_basis || exit 1;

  # The same, per-utterance.
steps/decode_basis_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_dev93_utt \
  exp/tri3b/decode${lang_suffix}_tgpr_dev93_basis_utt || exit 1;
steps/decode_basis_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3b/graph${lang_suffix}_tgpr data/test_eval92_utt \
  exp/tri3b/decode${lang_suffix}_tgpr_eval92_basis_utt || exit 1;


