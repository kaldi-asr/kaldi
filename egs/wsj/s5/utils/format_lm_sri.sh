#!/bin/bash

# Copyright 2012  Arnab Ghoshal
# Copyright 2010-2011  Microsoft Corporation

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

# Begin configuration section.
srilm_opts="-subset -prune-lowprobs -unk -tolower"
# end configuration sections

help_message="Usage: "`basename $0`" [options] lang_dir LM lexicon out_dir
Convert ARPA-format language models to FSTs. Change the LM vocabulary using SRILM.\n
options: 
  --help                 # print this message and exit
  --srilm-opts STRING    # options to pass to SRILM tools (default: '$srilm_opts')
";

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

lang_dir=$1
lm=$2
lexicon=$3
out_dir=$4
mkdir -p $out_dir

[ -f ./path.sh ] && . ./path.sh

loc=`which change-lm-vocab`
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/../change-lm-vocab ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir:$sdir/..
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi

echo "Converting '$lm' to FST"
tmpdir=$(mktemp -d kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

for f in phones.txt words.txt L.fst L_disambig.fst phones/; do
  cp -r $lang_dir/$f $out_dir || exit 1;
done

lm_base=$(basename $lm '.gz')
gunzip -c $lm | utils/find_arpa_oovs.pl $out_dir/words.txt \
  > $out_dir/oovs_${lm_base}.txt || exit 1;

# Removing all "illegal" combinations of <s> and </s>, which are supposed to 
# occur only at being/end of utt.  These can cause determinization failures 
# of CLG [ends up being epsilon cycles].
gunzip -c $lm \
  | egrep -v '<s> <s>|</s> <s>|</s> </s>' \
  | gzip -c > $tmpdir/lm.gz || exit 1;

awk '{print $1}' $out_dir/words.txt > $tmpdir/voc || exit 1;

# Change the LM vocabulary to be the intersection of the current LM vocabulary
# and the set of words in the pronunciation lexicon. This also renormalizes the 
# LM by recomputing the backoff weights, and remove those ngrams whose 
# probabilities are lower than the backed-off estimates.
change-lm-vocab -vocab $tmpdir/voc -lm $tmpdir/lm.gz -write-lm $tmpdir/out_lm \
  $srilm_opts || exit 1;

arpa2fst $tmpdir/out_lm | fstprint \
  | utils/eps2disambig.pl | utils/s2eps.pl \
  | fstcompile --isymbols=$out_dir/words.txt --osymbols=$out_dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
  | fstrmepsilon > $out_dir/G.fst || exit 1;

fstisstochastic $out_dir/G.fst

# The output is like:
# 9.14233e-05 -0.259833
# we do expect the first of these 2 numbers to be close to zero (the second is
# nonzero because the backoff weights make the states sum to >1).

# Everything below is only for diagnostic.
# Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
# this might cause determinization failure of CLG.
# #0 is treated as an empty word.
mkdir -p $out_dir/tmpdir.g
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} 
     END{print "0 0 #0 #0"; print "0";}' \
     < "$lexicon" > $out_dir/tmpdir.g/select_empty.fst.txt || exit 1;

fstcompile --isymbols=$out_dir/words.txt --osymbols=$out_dir/words.txt \
  $out_dir/tmpdir.g/select_empty.fst.txt \
  | fstarcsort --sort_type=olabel \
  | fstcompose - $out_dir/G.fst > $out_dir/tmpdir.g/empty_words.fst || exit 1;

fstinfo $out_dir/tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' \
  && echo "Language model has cycles with empty words" && exit 1

rm -r $out_dir/tmpdir.g


echo "Succeeded in formatting LM: '$lm'"
