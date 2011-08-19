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

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  spk2utt
#  utt2spk
#  text
# It creates a subset of that data, consisting of some specified
# number of utterances.  (The selected utterances are distributed
# evenly throughout the file, by the program ./subset_scp.pl).


if [ $# != 3 ]; then
  echo "Usage: subset_data_dir.sh <srcdir> <num-utt> <destdir>"
fi

srcdir=$1
numutt=$2
destdir=$3

if [ ! -f $srcdir/feats.scp ]; then
  echo "subset_data_dir.sh: no such file $srcdir/feats.scp" 
  exit 1;
fi

if [ $numutt -gt `cat $srcdir/feats.scp | wc -l` ]; then
  echo "subset_data_dir.sh: cannot subset to more utterances than you originally had."
  exit 1;
fi 


mkdir -p $destdir || exit 1;

# create feats.scp
scripts/subset_scp.pl $numutt $srcdir/feats.scp > $destdir/feats.scp || exit 1;

if [ -f $srcdir/wav.scp ]; then
  scripts/filter_scp.pl $destdir/feats.scp $srcdir/wav.scp > $destdir/wav.scp || exit 1;
else
  rm $destdir/wav.scp 2>/dev/null
fi

if [ -f $srcdir/utt2spk ]; then
  scripts/filter_scp.pl $destdir/feats.scp $srcdir/utt2spk > $destdir/utt2spk|| exit 1;
  scripts/utt2spk_to_spk2utt.pl $destdir/utt2spk > $destdir/spk2utt || exit 1;
fi

if [ -f $srcdir/text ]; then
  scripts/filter_scp.pl $destdir/feats.scp $srcdir/text > $destdir/text || exit 1;
fi

echo "Created a $numutt-utterance subset of $srcdir and put it in $destdir."
