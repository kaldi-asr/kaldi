#!/bin/bash

# Copyright 2010-2011 Microsoft Corporation

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

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang/, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

for x in train_si284 eval_nov92 eval_nov93 dev_nov93; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp
  cp data/local/$x.txt data/$x/txt
  cp data/local/$x.spk2utt data/$x/spk2utt
  cp data/local/$x.utt2spk data/$x/utt2spk
  scripts/filter_scp.pl data/$x/spk2utt data/local/spk2gender.map > data/$x/spk2gender
done

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_wsj_mfcc
for x in eval_nov92 eval_nov93 dev_nov93 train_si284; do 
 steps/make_mfcc.sh data/$x exp/make_mfcc/$x $mfccdir 4
done

