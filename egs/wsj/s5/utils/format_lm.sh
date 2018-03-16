#!/usr/bin/env bash

# Copyright 2012  Arnab Ghoshal
#           2010-2011  Microsoft Corporation
#           2016-2018  Johns Hopkins University (author: Daniel Povey)

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

set -e

if [ $# -ne 4 ]; then
  echo "Usage: $0 <lang_dir> <arpa-LM> <lexicon> <out_dir>"
  echo "E.g.: $0 data/lang data/local/lm/foo.kn.gz data/local/dict/lexicon.txt data/lang_test"
  echo "Convert ARPA-format language models to FSTs.";
  exit 1;
fi

lang_dir=$1
lm=$2
lexicon=$3
out_dir=$4
mkdir -p $out_dir

[ -f ./path.sh ] && . ./path.sh

echo "Converting '$lm' to FST"

# the -ef test checks if  source and target directory
# are two different directories in the filesystem
# if they are the same, the section guarded by the test
# would be actually harmfull (deleting the phones/ subdirectory)
if [ -e $out_dir ] && [ ! $lang_dir -ef $out_dir ] ; then
  if [ -e $out_dir/phones ] ; then
    rm -r $out_dir/phones
  fi

  for f in phones.txt words.txt topo L.fst L_disambig.fst phones oov.int oov.txt; do
     cp -r $lang_dir/$f $out_dir
  done
fi

lm_base=$(basename $lm '.gz')
gunzip -c $lm \
  | arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$out_dir/words.txt - $out_dir/G.fst
set +e
fstisstochastic $out_dir/G.fst
set -e
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
     < "$lexicon" > $out_dir/tmpdir.g/select_empty.fst.txt

fstcompile --isymbols=$out_dir/words.txt --osymbols=$out_dir/words.txt \
  $out_dir/tmpdir.g/select_empty.fst.txt \
  | fstarcsort --sort_type=olabel \
  | fstcompose - $out_dir/G.fst > $out_dir/tmpdir.g/empty_words.fst

fstinfo $out_dir/tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' \
  && echo "Language model has cycles with empty words" && exit 1

rm -r $out_dir/tmpdir.g


echo "Succeeded in formatting LM: '$lm'"
