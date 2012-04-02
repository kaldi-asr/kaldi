#!/bin/bash -u

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

# To be safe we suggest running the recipe line by line. Otherwise
# comment out the following line
exit 1;

. path.sh
local/timit_data_prep.sh /ais/gobi2/speech/TIMIT || exit 1;
# local/timit_data_prep.sh /export/corpora5/LDC/LDC93S1 || exit 1;
local/timit_train_lms.sh data/local || exit 1 ;
local/timit_format_data.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfccs

for test in train test dev ; do
  steps/make_mfcc.sh data/$test exp/make_mfcc/$test $mfccdir 4
done

# train monophone system.
steps/train_mono.sh data/train data/lang exp/mono || exit 1;

scripts/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1;
echo "Decoding test datasets."
for test in dev test ; do
  steps/decode_deltas.sh exp/mono data/$test data/lang exp/mono/decode_$test &
done
wait
scripts/average_wer.sh exp/mono/decode_*/wer > exp/mono/wer || exit 1;

# Get alignments from monophone system.
echo "Creating training alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali_train || exit 1;
echo "Creating dev alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/dev data/lang exp/mono exp/mono_ali_dev || exit 1;
echo "Creating test alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/test data/lang exp/mono exp/mono_ali_test || exit 1;


