#!/bin/bash

# In case we want to reuse some file in $librispeech_dir/local
librispeech_dir=../../librispeech/s5/
data=/export/a05/dgalvez/

data_url=www.openslr.org/resources/TODO # TODO
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

# TODO(galv): Reconsider this
set -euo pipefail

# TODO(galv): Modify openslr.org to contain the minified training dataset.
# for part in dev-clean dev-other train-clean-5; do
#   local/download_and_untar.sh $data $data_url $part
# done

$librispeech_dir/local/download_lm.sh $lm_url data/local/lm

# format the data as Kaldi data directories
for part in dev-clean-2 train-clean-5; do
  # use underscore-separated names in data directories.
  $librispeech_dir/local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
done

$librispeech_dir/local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict_nosp

utils/prepare_lang.sh data/local/dict_nosp \
   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

$librispeech_dir/local/format_lms.sh --src-dir data/lang_nosp data/local/lm

# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
  data/lang_nosp data/lang_nosp_test_tglarge
utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
  data/lang_nosp data/lang_nosp_test_fglarge

mfccdir=mfcc
# spread the mfccs over various machines, as this data-set is quite large.
if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  utils/create_split_dir.pl /export/b{07,14,16,17}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
    $mfccdir/storage
fi

for part in dev_clean_2 train_clean_5; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
  steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

# Get the shortest 500 utterances first because those are more likely
# to have accurate alignments.
utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short

# TODO(galv): Is this too many jobs for a smaller dataset?
# train a monophone system
steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
  data/train_500short data/lang_nosp exp/mono
# TODO: Understand why we use lang_nosp here...

(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/mono exp/mono/graph_nosp_tgsmall
  for test in dev_clean_2; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
      data/$test exp/mono/decode_nosp_tgsmall_$test
  done
)&

steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp exp/mono exp/mono_ali_clean_5

# train a first delta + delta-delta triphone system on all utterances
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
  2000 10000 data/train_clean_5 data/lang_nosp exp/mono_ali_clean_5 exp/tri1

# decode using the tri1 model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri1 exp/tri1/graph_nosp_tgsmall
  for test in dev_clean_2; do
    steps/decode.sh --nj 5 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
      data/$test exp/tri1/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
  done
)&

steps/align_si.sh --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp exp/tri1 exp/tri1_ali_clean_5


# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
   data/train_clean_5 data/lang_nosp exp/tri1_ali_clean_5 exp/tri2b

# decode using the LDA+MLLT model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri2b exp/tri2b/graph_nosp_tgsmall
  for test in dev_clean_2; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
      data/$test exp/tri2b/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
  done
)&

# Align utts using the tri2b model
steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
  data/train_clean_5 data/lang_nosp exp/tri2b exp/tri2b_ali_clean_5

# Train tri3b, which is LDA+MLLT+SAT
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
  data/train_clean_5 data/lang_nosp exp/tri2b_ali_clean_5 exp/tri3b

# decode using the tri3b model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri3b exp/tri3b/graph_nosp_tgsmall
  for test in dev_clean_2; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
      exp/tri3b/graph_nosp_tgsmall data/$test \
      exp/tri3b/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
  done
)&

# align the entire train_clean_5 subset using the tri3b model
steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
  data/train_clean_5 data/lang_nosp \
  exp/tri3b exp/tri3b_ali_clean_5

# Don't finish until all background decoding jobs are finished.
wait
