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

# If you give the --per-spk option, it will attempt to select
# the supplied number of utterances for each speaker (typically
# you would supply a much smaller number in this case).

perspk=false
if [ "$1" == "--per-spk" ]; then
  perspk=true;
  shift;
fi

if [ $# != 3 ]; then
  echo "Usage: subset_data_dir.sh [--per-spk] <srcdir> <num-utt> <destdir>"
  exit 1;
fi

srcdir=$1
numutt=$2
destdir=$3


if [ ! -f $srcdir/feats.scp ]; then
  echo "subset_data_dir.sh: no such file $srcdir/feats.scp" 
  exit 1;
fi


## scripting note: $perspk evaluates to true or false
## so this becomes the command true or false.
if $perspk; then
  mkdir -p $destdir
  awk '{ n='$numutt'; printf("%s ",$1); skip=1; while(n*(skip+1) <= NF-1) { skip++; }
         for(x=2; x<=NF && x <= n*skip; x += skip) { printf("%s ", $x); } 
         printf("\n"); }' <$srcdir/spk2utt >$destdir/spk2utt
  scripts/spk2utt_to_utt2spk.pl < $destdir/spk2utt > $destdir/utt2spk
  scripts/filter_scp.pl $destdir/utt2spk <$srcdir/feats.scp >$destdir/feats.scp
  [ -f $srcdir/wav.scp ] && scripts/filter_scp.pl $destdir/feats.scp <$srcdir/wav.scp >$destdir/wav.scp
  [ -f $srcdir/text ] && scripts/filter_scp.pl $destdir/feats.scp <$srcdir/text >$destdir/text
  [ -f $srcdir/spk2gender ] && scripts/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender
  srcutts=`cat $srcdir/utt2spk | wc -l`
  destutts=`cat $destdir/utt2spk | wc -l`
  echo "Retained $numutt utterances per speaker from data-dir $srcdir and put it in $destdir, reducing #utt from $srcutts to $destutts"
  exit 0;
else
  if [ $numutt -gt `cat $srcdir/feats.scp | wc -l` ]; then
    echo "subset_data_dir.sh: cannot subset to more utterances than you originally had."
    exit 1;
  fi 

  mkdir -p $destdir || exit 1;

  # create feats.scp
  scripts/subset_scp.pl $numutt $srcdir/feats.scp > $destdir/feats.scp || exit 1;
 
  if [ -f $srcdir/wav.scp ]; then
    scripts/filter_scp.pl $destdir/feats.scp $srcdir/mfc.scp > $destdir/mfc.scp || exit 1;
  else
    rm $destdir/mfc.scp 2>/dev/null
  fi

  if [ -f $srcdir/utt2spk ]; then
    scripts/filter_scp.pl $destdir/feats.scp $srcdir/utt2spk > $destdir/utt2spk|| exit 1;
    scripts/utt2spk_to_spk2utt.pl $destdir/utt2spk > $destdir/spk2utt || exit 1;
  fi

  [ -f $srcdir/text ] && scripts/filter_scp.pl $destdir/feats.scp <$srcdir/text >$destdir/text

  [ -f $srcdir/spk2gender ] && scripts/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender

  echo "Created a $numutt-utterance subset of $srcdir and put it in $destdir."

  exit 0;
fi
