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


. utils/parse_options.sh

if [ $# -ne 4 ] && [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <lang-dir> <arpa-LM> [<lexicon>] <out-dir>"
  echo "The <lexicon> argument is no longer needed but is supported for back compatibility"
  echo "E.g.: utils/format_lm_sri.sh data/lang data/local/lm/foo.kn.gz data/local/dict/lexicon.txt data/lang_test"
  echo "Converts ARPA-format language models to FSTs. Change the LM vocabulary using SRILM."
  echo "Note: if you want to just convert ARPA LMs to FSTs, there is a simpler way to do this"
  echo "that doesn't require SRILM: see utils/format_lm.sh"
  echo "options:"
  echo " --help                 # print this message and exit"
  echo " --srilm-opts STRING      # options to pass to SRILM tools (default: '$srilm_opts')"
  exit 1;
fi


if [ $# -eq 4 ] ; then
  lang_dir=$1
  lm=$2
  lexicon=$3
  out_dir=$4
else
  lang_dir=$1
  lm=$2
  out_dir=$3
fi

mkdir -p $out_dir

for f in $lm $lang_dir/words.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected input file $f to exist."
    exit 1;
  fi
done

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
    echo or installed in $sdir.  cd to ../../../tools and run
    echo extras/install_srilm.sh.
    exit 1
  fi
fi

echo "Converting '$lm' to FST"
tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p $out_dir
cp -r $lang_dir/* $out_dir || exit 1;

lm_base=$(basename $lm '.gz')
awk '{print $1}' $out_dir/words.txt > $tmpdir/voc || exit 1;

# Change the LM vocabulary to be the intersection of the current LM vocabulary
# and the set of words in the pronunciation lexicon. This also renormalizes the
# LM by recomputing the backoff weights, and remove those ngrams whose
# probabilities are lower than the backed-off estimates.
change-lm-vocab -vocab $tmpdir/voc -lm $lm -write-lm - $srilm_opts | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$out_dir/words.txt - $out_dir/G.fst || exit 1

fstisstochastic $out_dir/G.fst

# The output is like:
# 9.14233e-05 -0.259833
# we do expect the first of these 2 numbers to be close to zero (the second is
# nonzero because the backoff weights make the states sum to >1).

echo "Succeeded in formatting LM '$lm' -> '$out_dir/G.fst'"
