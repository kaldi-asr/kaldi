#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script shows the steps needed to build a recognizer for certain languages
# of the GlobalPhone corpus. 
# !!! NOTE: The current recipe assumes that you have pre-built LMs. 
echo "This shell script may run as-is on your system, but it is recommended 
that you run the commands one by one by copying and pasting into the shell."
#exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

# INSTALLING REQUIRED TOOLS:
#  This recipe requires shorten and sox (we use shorten 3.6.1 and sox 14.3.2).
#  If you don't have them, use the install.sh script to install them.
( which shorten >&/dev/null && which sox >&/dev/null && \
  echo "shorten and sox found: you may want to edit the path.sh file." ) || \
  { echo "shorten and/or sox not found on PATH. Installing..."; install.sh; }

. path.sh || { echo "Cannot source path.sh"; exit 1; }

# Set the locations of the GlobalPhone corpus and language models
GP_CORPUS=/mnt/matylda2/data/GLOBALPHONE
GP_LM=/mnt/matylda6/ijanda/GLOBALPHONE_LM

# Set the languages that will actually be processed
# export GP_LANGUAGES="CZ FR GE PL PO RU SP VN"
export GP_LANGUAGES="CZ GE PL PO SP"

# The following data preparation step actually converts the audio files from 
# shorten to WAV to take out the empty files and those with compression errors. 
# local/gp_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$GP_CORPUS \
#   --languages="$GP_LANGUAGES"

for L in $GP_LANGUAGES; do
  utils/prepare_lang.sh --position-dependent-phones false \
    data/$L/local/dict "<UNK>" data/$L/local/lang_tmp data/$L/lang \
    >& data/$L/prepare_lang.log || exit 1;
done

# Convert the different available language models to FSTs, and create separate 
# decoding configurations for each.
local/gp_format_data.sh --filter-vocab-sri false $GP_LM $GP_LANGUAGES;
local/gp_format_data.sh --filter-vocab-sri true $GP_LM $GP_LANGUAGES;

# Now make MFCC features.
for L in $GP_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train dev eval; do
    ( steps/make_mfcc.sh --nj 6 --cmd "$train_cmd" data/$L/$x \
        exp/$L/make_mfcc/$x $mfccdir;
      steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir; ) &
  done
done
wait;

for L in $GP_LANGUAGES; do
  mkdir -p exp/$L/mono;
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono >& exp/$L/mono/train.log
#   # The following 3 commands will not run as written, since the LM directories
#   # will be different across sites. Edit the 'lang_test' to match what is 
#   # available
#   utils/mkgraph.sh --mono data/$L/lang_test_tgpr exp/$L/mono \
#     exp/$L/mono/graph
#   utils/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh \
#     exp/$L/mono/graph data/$L/dev exp/$L/mono/decode_dev
#   utils/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh \
#     exp/$L/mono/graph data/$L/eval exp/$L/mono/decode_eval
done


for L in $GP_LANGUAGES; do
  mkdir -p exp/$L/mono_ali
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono exp/$L/mono_ali \
    >& exp/$L/mono_ali/align.log

  num_states=$(grep "^$L" conf/tri.conf | cut -f2)
  num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
  mkdir -p exp/$L/tri1
  steps/train_deltas.sh --nj 10 --cmd "$train_cmd" \
    $num_states $num_gauss data/$L/train data/$L/lang exp/$L/mono_ali \
    exp/$L/tri1 >& exp/$L/tri1/train.log

#   # Like with the monophone systems, the following 3 commands will not run.
#   # Edit the 'lang_test' to match what is available.
#   utils/mkgraph.sh data/$L/lang_test_tgpr exp/$L/tri1 exp/$L/tri1/graph
#   utils/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh \
#     exp/$L/tri1/graph data/$L/dev exp/$L/tri1/decode_dev
#   utils/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh \
#     exp/$L/tri1/graph data/$L/eval exp/$L/tri1/decode
done

for L in $GP_LANGUAGES; do
  mkdir -p exp/$L/tri1_ali
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri1 exp/$L/tri1_ali \
    >& exp/$L/tri1_ali/align.log

  num_states=$(grep "^$L" conf/tri.conf | cut -f2)
  num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
  mkdir -p exp/$L/tri2a
  steps/train_deltas.sh --nj 10 --cmd "$train_cmd" \
    $num_states $num_gauss data/$L/train data/$L/lang exp/$L/tri1_ali \
    exp/$L/tri2a >& exp/$L/tri2a/train.log
done

