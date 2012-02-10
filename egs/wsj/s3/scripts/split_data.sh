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

split_per_spk=true
if [ "$1" == "--per-utt" ]; then
  split_per_spk=false
  shift
fi


if [ $# != 2 ]; then
  echo "Usage: split_data.sh data-dir num-to-split"
  exit 1
fi

data=$1
numsplit=$2

if [ $numsplit -le 0 ]; then
  echo "Invalid num-split argument $numsplit";
  exit 1;
fi

n=0;
feats=""
wavs=""
utt2spks=""
texts=""

nu=`cat $data/utt2spk | wc -l`
nf=`cat $data/feats.scp | wc -l`
nt=`cat $data/text | wc -l`
if [ $nu -ne $nf ]; then
  echo "split_data.sh: warning, #lines is (utt2spk,feats.scp) is ($nu,$nf); this script "
  echo " may produce incorrectly split data."
  echo "use scripts/fix_data_dir.sh to fix this."
fi
if [ $nt -ne 0 -a $nu -ne $nt ]; then
  echo "split_data.sh: warning, #lines is (utt2spk,text) is ($nu,$nt); this script "
  echo " may produce incorrectly split data."
  echo "use scripts/fix_data_dir.sh to fix this."
fi

# `scripts/get_split.pl` returns "0 1 2 3" or "00 01 .. 18 19" or whatever.
for n in `scripts/get_splits.pl $numsplit`; do
   mkdir -p $data/split$numsplit/$n
   feats="$feats $data/split$numsplit/$n/feats.scp"
   wavs="$wavs $data/split$numsplit/$n/wav.scp"
   texts="$texts $data/split$numsplit/$n/text"
   utt2spks="$utt2spks $data/split$numsplit/$n/utt2spk"
done


if $split_per_spk; then
  utt2spk_opt="--utt2spk=$data/utt2spk"
else
  utt2spk_opt=
fi

scripts/split_scp.pl $utt2spk_opt $data/utt2spk $utt2spks || exit 1

scripts/split_scp.pl $utt2spk_opt $data/feats.scp $feats || exit 1
[ -f $data/wav.scp ] && \
  scripts/split_scp.pl $utt2spk_opt $data/wav.scp $wavs
[ -f $data/text ] && \
 scripts/split_scp.pl $utt2spk_opt $data/text $texts

for n in `scripts/get_splits.pl $numsplit`; do
   scripts/utt2spk_to_spk2utt.pl $data/split$numsplit/$n/utt2spk > $data/split$numsplit/$n/spk2utt || exit 1;
   # for completeness, also split the spk2gender file
   [ -f $data/spk2gender ] && \
     scripts/filter_scp.pl $data/split$numsplit/$n/spk2utt $data/spk2gender > $data/split$numsplit/$n/spk2gender 
done

exit 0
