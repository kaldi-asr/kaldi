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
exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

# INSTALLING REQUIRED TOOLS:
#  This recipe requires shorten and sox (we use shorten 3.6.1 and sox 14.3.2).
#  If you don't have them, use the install.sh script to install them.
( which shorten >&/dev/null && which sox >&/dev/null && \
  ehco "shorten and sox found: you may want to edit the path.sh file." ) || \
  { echo "shorten and/or sox not found on PATH. Installing..."; 
    install.sh }

# Set the locations of the GlobalPhone corpus and language models
GP_CORPUS=/mnt/matylda2/data/GLOBALPHONE
GP_LM=/mnt/matylda6/ijanda/GLOBALPHONE_LM

# Set the languages that will actually be processed
# export GP_LANGUAGES="CZ FR GE PL PO RU SP VN"
export GP_LANGUAGES="PO SP"

# The following data preparation step actually converts the audio files from 
# shorten to WAV to take out the empty files and those with compression errors. 
local/gp_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$GP_CORPUS

local/gp_format_data.sh --hmm-proto=conf/topo.proto --work-dir=$PWD

# Now make MFCC features.
for LCODE in $GP_LANGUAGES; do
  mfccdir=$WORKDIR/MFCC/$LCODE
  for x in train dev eval; do
    steps/make_mfcc.sh --num-jobs 6 --cmd "$train_cmd" data/$LCODE/$x \
      exp/$LCODE/make_mfcc/$x $mfccdir
  done
done

for LCODE in $GP_LANGUAGES; do
  steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
    data/$LCODE/train data/$LCODE/lang exp/$LCODE/mono
  # The following 3 commands will not run as written, since the LM directories
  # will be different across sites. Edit the 'lang_test' to match what is 
  # available
  utils/mkgraph.sh --mono data/$LCODE/lang_test exp/$LCODE/mono \
    exp/$LCODE/mono/graph
  utils/decode.sh --qcmd "$decode_cmd" steps/decode_deltas.sh \
    exp/$LCODE/mono/graph data/$LCODE/dev exp/$LCODE/mono/decode_dev
  utils/decode.sh --qcmd "$decode_cmd" steps/decode_deltas.sh \
    exp/$LCODE/mono/graph data/$LCODE/eval exp/$LCODE/mono/decode_eval
done


# This queue option will be supplied to all alignment
# and training scripts.  Note: you have to supply the same num-jobs
# to the alignment and training scripts, as the archives are split
# up in this way.

for LCODE in $GP_LANGUAGES; do
  steps/align_deltas.sh --num-jobs 10 --qcmd "$train_cmd" \
    data/$LCODE/train data/$LCODE/lang exp/$LCODE/mono exp/$LCODE/mono_ali

  steps/train_deltas.sh --num-jobs 10 --qcmd "$train_cmd" \
    2000 10000 data/$LCODE/train data/$LCODE/lang exp/$LCODE/mono_ali \
    exp/$LCODE/tri1

  # Like with the monophone systems, the following 3 commands will not run.
  # Edit the 'lang_test' to match what is available.
  utils/mkgraph.sh data/$LCODE/lang_test exp/$LCODE/tri1 exp/$LCODE/tri1/graph
  utils/decode.sh --qcmd "$decode_cmd" steps/decode_deltas.sh \
    exp/$LCODE/tri1/graph data/$LCODE/dev exp/$LCODE/tri1/decode_dev
  utils/decode.sh --qcmd "$decode_cmd" steps/decode_deltas.sh \
    exp/$LCODE/tri1/graph data/$LCODE/eval exp/$LCODE/tri1/decode

done