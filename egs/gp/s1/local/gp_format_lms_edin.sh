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
  local test=$work_dir/test_${lm_suffix}

  mkdir -p $test
  for f in phones.txt words.txt phones_disambig.txt L.fst L_disambig.fst \
           silphones.csl nonsilphones.csl; do
    cp $work_dir/lang_test/$f $test
  done

  # kkm: I am removing fstdeterminizelog from the following pipe, no point.
  gunzip -c $work_dir/local/lm_${lm_suffix}.arpa.gz \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst
  set +e
  fstisstochastic $test/G.fst
  set -e
  # The output is like:
  # 9.14233e-05 -0.259833
  # we do expect the first of these 2 numbers to be close to zero (the second is
  # nonzero because the backoff weights make the states sum to >1).
  # Because of the <s> fiasco for these particular LMs, the first number is not
  # as close to zero as it could be.

  # Everything below is only for diagnostic.
  # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
  # this might cause determinization failure of CLG.
  # #0 is treated as an empty word.
  mkdir -p tmpdir.g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < $work_dir/local/lexicon_??.txt  >tmpdir.g/select_empty.fst.txt
  fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt tmpdir.g/select_empty.fst.txt | \
   fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > tmpdir.g/empty_words.fst
  fstinfo tmpdir.g/empty_words.fst | grep cyclic | grep -w 'y' &&
    echo "Language model has cycles with empty words" && exit 1
  rm -r tmpdir.g

}

PROG=`basename $0`;
usage="Usage: $PROG data_dir\n
 Convert ARPA-format language models to FSTs for GlobalPhone langauges.\n
 (Currently converts for German, Portuguese, Spanish & Swedish).\n";

if [ $# -ne 1 ]; then
  error_exit $usage;
fi
WDIR=`read_dirname $1`;

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test directory.

echo "Preparing language models for test"

# German - 17K
{ format_lms GE17k_bg $WDIR/GE;
  format_lms GE17k_tg $WDIR/GE;
  format_lms GE17k_tg_pr $WDIR/GE; } >& $WDIR/GE/format_lms.log

# German - 60K
{ format_lms GE60k_bg $WDIR/GE;
  format_lms GE60k_tg $WDIR/GE;
  format_lms GE60k_tg_pr $WDIR/GE; } >> $WDIR/GE/format_lms.log 2>&1

# Portuguese - 60K
{ format_lms PO60k_bg $WDIR/PO;
  format_lms PO60k_tg $WDIR/PO;
  format_lms PO60k_tg_pr $WDIR/PO; } >& $WDIR/PO/format_lms.log

# Spanish - 23K
{ format_lms SP23k_bg $WDIR/SP;
  format_lms SP23k_tg $WDIR/SP;
  format_lms SP23k_tg_pr $WDIR/SP; } >& $WDIR/SP/format_lms.log

# Swedish - 24K
# TODO(arnab): Something going wrong with the Swedish trigram LM.
{ # format_lms SW24k_tg $WDIR/SW;
  # format_lms SW24k_tg_pr $WDIR/SW;
  format_lms SW24k_bg $WDIR/SW; } >& $WDIR/SW/format_lms.log

echo "Preparing test data"
