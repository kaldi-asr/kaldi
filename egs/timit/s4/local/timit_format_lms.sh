#!/bin/bash -u

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

set -o errexit
#set -o pipefail

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function read_dirname () {
  [ -d "$1" ] || error_exit "Argument '$1' not a directory";
  local retval=`cd $1 2>/dev/null && pwd || exit 1`
  echo $retval
}

function format_lms () {
  local lm_suffix=$1;
  local work_dir=$2
  local test=$work_dir/lang_test_${lm_suffix}

  mkdir -p $test
  for f in phones.txt words.txt phones_disambig.txt L.fst L_disambig.fst \
           silphones.csl nonsilphones.csl; do
    cp $work_dir/lang_test/$f $test
  done

  # Removing all "illegal" combinations of <s> and </s>, which are supposed to 
  # occur only at being/end of utt.  These can cause determinization failures 
  # of CLG [ends up being epsilon cycles].
  gunzip -c $work_dir/local/lm_${lm_suffix}.arpa.gz \
    | egrep -v '<s> <s>|</s> <s>|</s> </s>' \
    | arpa2fst - | fstprint \
    | eps2disambig.pl | s2eps.pl \
    | fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt \
      --keep_isymbols=false --keep_osymbols=false \
    | fstrmepsilon > $test/G.fst
  set +e
  fstisstochastic $test/G.fst
  set -e
}

PROG=`basename $0`;
usage="Usage: $PROG data_dir\n
 Convert ARPA-format language models to FSTs.\n";

if [ $# -ne 1 ]; then
  error_exit $usage;
fi
WDIR=`read_dirname $1`;

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test directory.

echo "Preparing language models for test"
format_lms phone_bg $WDIR >& $WDIR/format_lms.log
