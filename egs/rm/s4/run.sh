#!/bin/bash

# Copyright 2010-2011 Microsoft Corporation
# Copyright 2012 Vassil Panayotov

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

# It is recommended that you do not invoke this file from the shell, but
# run the paths one by one, by hand.

# IMPORTANT:
# This script file cannot be run as-is; some paths in it need to be changed
# before you can run it. Please edit the path variables in ./path.sh
source ./path.sh

# First step is to do data preparation: 
# This just creates some text files, it is fast.
cd data_prep
./run.sh $RM1_ROOT
cd ..

mkdir -p data

# This next step converts the lexicon, grammar, etc., into FST format.
steps/prepare_graphs.sh

# The models and decoding results are stored under "exp" directory.

# Convert Sphinx feature files to Kaldi tables
mfccdir=./mfcc
steps/make_mfcc.sh $mfccdir train
steps/make_mfcc.sh $mfccdir test

# Monophone training and decoding
steps/train_mono.sh
steps/decode_mono.sh 

# Triphone training and decoding
steps/train_tri1.sh
steps/decode_tri1.sh
steps/train_tri2a.sh
steps/decode_tri2a.sh

echo "-- Done!"
exit 0
