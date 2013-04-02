#!/bin/bash

. cmd.sh

mfccdir=mfcc

# Make "per-utterance" versions of the test sets where the speaker
# information corresponds to utterances-- to demonstrate adaptation on
# short utterances, particularly for basis fMLLR
for x in "test" ; do
  y=${x}_utt
  rm -r data/$y
  cp -r data/$x data/$y
  cat data/$x/utt2spk | awk '{print $1, $1;}' > data/$y/utt2spk;
  cp data/$y/utt2spk data/$y/spk2utt;
  steps/compute_cmvn_stats.sh data/$y exp/make_mfcc/$y $mfccdir || exit 1; 
done

 # basis fMLLR experiments.
 # First a baseline: decode per-utterance with normal fMLLR.
steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3b/graph_bg data/test_utt exp/tri3b/decode_bg_test_utt || exit 1;

 # get the fMLLR basis.
steps/get_fmllr_basis.sh --cmd "$train_cmd" data/train data/lang exp/tri3b

 # decoding tri3b with basis fMLLR
steps/decode_basis_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3b/graph_bg data/test exp/tri3b/decode_bg_test_basis || exit 1;

  # The same, per-utterance.
steps/decode_basis_fmllr.sh --nj 30 --cmd "$decode_cmd" \
  exp/tri3b/graph_bg data/test_utt exp/tri3b/decode_bg_test_basis_utt || exit 1;



