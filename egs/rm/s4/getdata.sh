#!/bin/bash

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

source path.sh

# Download and extract CMU's feature files
mkdir -p $RM1_ROOT
wget -P $RM1_ROOT http://www.speech.cs.cmu.edu/databases/rm1/rm1_cepstra.tar.gz ||
 wget -P $RM1_ROOT http://sourceforge.net/projects/kaldi/files/rm1_cepstra.tar.gz
tar -C $RM1_ROOT/ -xf $RM1_ROOT/rm1_cepstra.tar.gz

# Download the G.fst graph produced from 'wp_gram.txt'
wget -P $RM1_ROOT http://sourceforge.net/projects/kaldi/files/RM_G.fst