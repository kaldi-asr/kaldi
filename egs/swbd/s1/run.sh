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

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you

# might want to run this script on a machine that has plenty of memory.

# To get the CMU dictionary, do:
svn co https://cmusphinx.svn.sourceforge.net/svnroot/cmusphinx/trunk/cmudict/
# got this at revision 10966 in the last tests done before releasing v1.0.
# can add -r 10966 for strict compatibility.


# Data prep
local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2

local/swbd_p1_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=/mnt/matylda6/ijanda/kaldi_swbd_mfcc

steps/make_mfcc_segs.sh data/train exp/make_mfcc/train $mfccdir 8
