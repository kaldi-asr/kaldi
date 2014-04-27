#!/bin/bash
# Copyright 2010-2013 Microsoft Corporation 
#                     Johns Hopkins University (Author: Daniel Povey)

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
  echo "Usage: split_data.sh <data-dir> <num-to-split>"
  echo "This script will not split the data-dir if it detects that the output is newer than the input."
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
nt=`cat $data/text 2>/dev/null | wc -l` # take it as zero if no such file
if [ $nu -ne $nf ]; then
  echo "split_data.sh: warning, #lines is (utt2spk,feats.scp) is ($nu,$nf); this script "
  echo " may produce incorrectly split data."
  echo "use utils/fix_data_dir.sh to fix this."
fi
if [ $nt -ne 0 -a $nu -ne $nt ]; then
  echo "split_data.sh: warning, #lines is (utt2spk,text) is ($nu,$nt); this script "
  echo " may produce incorrectly split data."
  echo "use utils/fix_data_dir.sh to fix this."
fi

s1=$data/split$numsplit/1
if [ ! -d $s1 ]; then 
  need_to_split=true
else 
  need_to_split=false
  for f in utt2spk spk2utt spk2warp feats.scp text wav.scp cmvn.scp spk2gender \
    vad.scp segments reco2file_and_channel utt2lang; do
    if [[ -f $data/$f && ( ! -f $s1/$f || $s1/$f -ot $data/$f ) ]]; then
      need_to_split=true
    fi
  done
fi

if ! $need_to_split; then
  exit 0;
fi
  
for n in `seq $numsplit`; do
   mkdir -p $data/split$numsplit/$n
   feats="$feats $data/split$numsplit/$n/feats.scp"
   vads="$vads $data/split$numsplit/$n/vad.scp"
   texts="$texts $data/split$numsplit/$n/text"
   utt2spks="$utt2spks $data/split$numsplit/$n/utt2spk"
   utt2langs="$utt2langs $data/split$numsplit/$n/utt2lang"
done

if $split_per_spk; then
  utt2spk_opt="--utt2spk=$data/utt2spk"
else
  utt2spk_opt=
fi

utils/split_scp.pl $utt2spk_opt $data/utt2spk $utt2spks || exit 1

utils/split_scp.pl $utt2spk_opt $data/feats.scp $feats || exit 1

[ -f $data/text ] && utils/split_scp.pl $utt2spk_opt $data/text $texts

[ -f $data/vad.scp ] && utils/split_scp.pl $utt2spk_opt $data/vad.scp $vads

[ -f $data/utt2lang ] && utils/split_scp.pl $utt2spk_opt $data/utt2lang $utt2langs

# If lockfile is not installed, just don't lock it.  It's not a big deal.
which lockfile >&/dev/null && lockfile -l 60 $data/.split_lock 

for n in `seq $numsplit`; do
   dsn=$data/split$numsplit/$n
   utils/utt2spk_to_spk2utt.pl $dsn/utt2spk > $dsn/spk2utt || exit 1;
   for f in spk2gender spk2warp cmvn.scp; do
     [ -f $data/$f ] && \
       utils/filter_scp.pl $dsn/spk2utt $data/$f > $dsn/$f
   done
   if [ -f $data/segments ]; then
     utils/filter_scp.pl $dsn/utt2spk $data/segments > $dsn/segments
      awk '{print $2;}' $dsn/segments |sort|uniq > $data/tmp.reco # recording-ids.
     [ -f $data/reco2file_and_channel ] &&
     utils/filter_scp.pl $data/tmp.reco $data/reco2file_and_channel > $dsn/reco2file_and_channel
     [ -f $data/wav.scp ] && utils/filter_scp.pl $data/tmp.reco $data/wav.scp  > $dsn/wav.scp
     rm $data/tmp.reco
   else # else wav indexed by utterance -> filter on this.
     [ -f $data/wav.scp ] &&
       utils/filter_scp.pl $dsn/utt2spk $data/wav.scp > $dsn/wav.scp
   fi
done

rm -f $data/.split_lock

exit 0
