#!/bin/bash
#
# This recipe uses WSJ models and TED-LIUM audio with un-aligned transcripts.
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#
# Apache 2.0
#

. cmd.sh
. path.sh


set -e -o pipefail -u

nj=35
decode_nj=30   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.

. utils/parse_options.sh # accept options

# Data preparation
local/download_data.sh

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B
local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

local/wsj_format_data.sh

local/prepare_data.sh

# Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
# lets us use more jobs for decoding etc.
# [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
#  more than our normal 30 jobs.]
for dset in dev test train; do
  utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
done
  
local/train_lm.sh

local/prepare_dict.sh --dict-suffix "_nosp" \
  data/local/local_lm/data/work/wordlist

utils/prepare_lang.sh data/local/dict_nosp \
  "<unk>" data/local/lang_nosp data/lang_nosp

local/format_lms.sh

# Feature extraction
for set in train_si284; do
  dir=data/$set
  steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" $dir
  steps/compute_cmvn_stats.sh $dir
  utils/fix_data_dir.sh $dir
done

utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/wsj_mono0a || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_si84_half data/lang_nosp exp/wsj_mono0a exp/wsj_mono0a_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
  data/train_si84_half data/lang_nosp exp/wsj_mono0a_ali exp/wsj_tri1 || exit 1;

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84 data/lang_nosp exp/wsj_tri1 exp/wsj_tri1_ali_si84 || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
  data/train_si84 data/lang_nosp exp/wsj_tri1_ali_si84 exp/wsj_tri2b || exit 1;

# Align tri2b system with si84 data.
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_si84 \
  data/lang_nosp exp/wsj_tri2b exp/wsj_tri2b_ali_si84  || exit 1;

# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
  data/train_si84 data/lang_nosp exp/wsj_tri2b_ali_si84 exp/wsj_tri3b || exit 1;

# From 3b system, align all si284 data.
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_si284 data/lang_nosp exp/wsj_tri3b exp/wsj_tri3b_ali_si284 || exit 1;

# From 3b system, train another SAT system (tri4a) with all the si284 data.
steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
  data/train_si284 data/lang_nosp exp/wsj_tri3b_ali_si284 exp/wsj_tri4a || exit 1;

utils/mkgraph.sh data/lang_nosp exp/wsj_tri4a exp/wsj_tri4a/graph_nosp

(
for dset in dev test; do
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
    exp/wsj_tri4a/graph_nosp data/${dset} exp/wsj_tri4a/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
    data/${dset} exp/wsj_tri4a/decode_nosp_${dset} exp/wsj_tri4a/decode_nosp_${dset}_rescore
done
) &

wait 

echo "$0: success."
exit 0
