#!/bin/bash

## For this setup we don't have a large enough number of speakers for it to make
## sense to compute the basis for basis-fMLLR using the actual speakers, since
## we'd need at least 40*(40+1) = 1640 speakers.  So we do it per utterance.


. cmd.sh

mfccdir=mfcc

# Make "per-utterance" versions of the train and test sets
for x in train test; do
  y=${x}_utt
  rm -r data/$y
  cp -r data/$x data/$y
  cat data/$x/utt2spk | awk '{print $1, $1;}' > data/$y/utt2spk;
  cp data/$y/utt2spk data/$y/spk2utt;
  steps/compute_cmvn_stats.sh data/$y exp/make_mfcc/$y $mfccdir || exit 1; 
done


# Get fMLLR basis for tri3b (output in exp/tri3b/fmllr.basis)
steps/get_fmllr_basis.sh --cmd "$train_cmd" data/train data/lang exp/tri3b

# get the online_alimdl (exp/tri3b/final.online_alimdl)
steps/online/get_alimdl.sh --cmd "$train_cmd" --cmvn-sliding-config conf/cmvn_sliding.conf data/train exp/tri3b
