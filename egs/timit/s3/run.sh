.#!/bin/bash -u

# Copyright 2012  Navdeep Jaitly

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

exit 1;

. path.sh
local/timit_data_prep.sh /ais/gobi2/speech/TIMIT
local/timit_train_lms.sh data/local
local/timit_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfccs

for test in train test dev ; do
  steps/make_mfcc.sh data/$test exp/make_mfcc/$test $mfccdir 4
done

# train monophone system.
steps/train_mono.sh data/train data/lang exp/mono

scripts/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph
echo "Decoding test datasets."
for test in dev test ; do
  steps/decode_deltas.sh exp/mono data/$test data/lang exp/mono/decode_$test &
done
wait
scripts/average_wer.sh exp/mono/decode_*/wer > exp/mono/wer

# Get alignments from monophone system.
echo "Creating training alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali_train
echo "Creating dev alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/dev data/lang exp/mono exp/mono_ali_dev
echo "Creating test alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/test data/lang exp/mono exp/mono_ali_test


