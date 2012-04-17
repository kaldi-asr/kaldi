#!/bin/bash

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


source ./path.sh

# call the next line with the directory where the RM data is
local/rm_data_prep.sh $RM1_ROOT || exit 1;

local/rm_format_data.sh || exit 1;

# the directory, where you want to store MFCC features.
featdir=data/rm_feats

# convert the Sphinx feature files to Kaldi tables
for x in train test; do
 steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir  || exit 1;
done

scripts/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;

# train monophone system.
steps/train_mono.sh data/train.1k data/lang exp/mono  || exit 1;

# monophone decoding
local/decode.sh --mono steps/decode_deltas.sh exp/mono/decode || exit 1;

# Get alignments from monophone system.
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
local/decode.sh steps/decode_deltas.sh exp/tri1/decode || exit 1;

# align tri1
steps/align_deltas.sh --graphs "ark,s,cs:gunzip -c exp/tri1/graphs.fsts.gz|" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2a [delta+delta-deltas]
steps/train_deltas.sh data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

# decode tri2a
local/decode.sh steps/decode_deltas.sh exp/tri2a/decode || exit 1;
