#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)
# Copyright 2016 KTH Royal Institute of Technology (Author: Emelie Kullmann)

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

KALDI_ROOT=$(pwd)/../../..

exproot=$(pwd)
dir=data/local/dict
mkdir -p $dir

# Dictionary preparation:
# This lexicon was created using eSpeak. 
# To extend the setup, see local/dict_prep.sh

# Copy pre-made phone table and questions file
cp local/dictsrc/nonsilence_phones.txt $dir/nonsilence_phones.txt
cp local/dictsrc/extra_questions.txt $dir/extra_questions.txt
cp local/dictsrc/silence_phones.txt $dir/silence_phones.txt
cp local/dictsrc/optional_silence.txt $dir/optional_silence.txt


# Copy pre-made lexicon
wget http://www.openslr.org/resources/29/lexicon-sv.tgz --directory-prefix=data/local/data/download
tar -xzf data/local/data/download/lexicon-sv.tgz -C $dir



echo "Dictionary preparation succeeded"

