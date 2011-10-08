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



# Data prep
local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2

local/swbd_p1_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
#mfccdir=/mnt/matylda6/ijanda/kaldi_swbd_mfcc
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_swbd_mfcc
cmd="queue.pl -q all.q@@blade" # remove the option if no queue.
local/make_mfcc_segs.sh --num-jobs 10 --cmd "$cmd" data/train exp/make_mfcc/train $mfccdir

# Now-- there are 264k utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.
scripts/subset_data_dir.sh --shortest data/train 100000 data/train_100kshort
scripts/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
local/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

( . path.sh;
  cp data/train_10k_nodup/feats.scp{,.bak} 
  mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_swbd_mfcc
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_10k_nodup.ark,$mfccdir/kaldi_swbd_10k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_10k_nodup.scp data/train_10k_nodup/feats.scp
)
  
