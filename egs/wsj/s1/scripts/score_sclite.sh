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


# Does the sclite version of scoring in decode directories.

if [ $# != 1 ]; then
   echo "Usage: scripts/score_sclite.sh <decode-dir>"
   exit 1;
fi

sclite=../tools/sctk-2.4.0/bin/sclite

if [ ! -f $sclite  ]; then
   echo "The sclite program is not there.  Follow the INSTALL instructions in ../tools";
   exit 1;
fi

dir=$1

scoredir=$dir/scoring
mkdir $scoredir

cat $dir/test?*.tra  | \
  scripts/int2sym.pl --ignore-first-field data/words.txt | \
  sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
  scripts/transcript2hyp.pl > $scoredir/hyp

cat data/test_trans.txt | scripts/transcript2hyp.pl | sed 's:<NOISE>::g' | \
  sed 's:<SPOKEN_NOISE>::g' > $scoredir/ref

$sclite -r $scoredir/ref trn -h $scoredir/hyp trn -i wsj -o all -o dtl

