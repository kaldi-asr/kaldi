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
set -o pipefail

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || error_exit "Argument '$dir_name' not a directory";
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <arguments>\n
Prepare train, dev, eval file lists for a language.\n\n
Required arguments:\n
  --hmm-proto=FILE\tPrototype of the HMM topology\n
  --work-dir=DIR\t\tWorking directory\n
";

if [ $# -lt 2 ]; then
  error_exit $usage;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --hmm-proto=*)
  PROTO=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -f $PROTO ] || error_exit "Cannot find HMM prototype file '$PROTO'"; 
  shift ;;
  --work-dir=*)
  WDIR=`read_dirname $1`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; error_exit $usage ;;
  esac
done

cd $WDIR
. ./path.sh

echo "Preparing train data"

for LCODE in GE PO SP SW; do
# (0) Create a directory to contain files needed in training:
  for x in train dev eval; do 
    mkdir -p data/$LCODE/$x
    cp data/$LCODE/local/${x}_${LCODE}_wav.scp data/$LCODE/$x/wav.scp
    cp data/$LCODE/local/${x}_${LCODE}.trans2 data/$LCODE/$x/text
    cp data/$LCODE/local/${x}_${LCODE}.spk2utt data/$LCODE/$x/spk2utt
    cp data/$LCODE/local/${x}_${LCODE}.utt2spk data/$LCODE/$x/utt2spk
  done

  mkdir -p data/$LCODE/lang
  cp data/$LCODE/local/phones.txt -t data/$LCODE/lang/
  cp data/$LCODE/local/words.txt -t data/$LCODE/lang/

# (1) Generate colon-separated lists of silence and non-silence phones, and 
#     the file 'oov.txt' containing a word that all OOVs map to during training.
  silphones="SIL SPN";
  silphones.pl data/$LCODE/lang/phones.txt "$silphones" \
    data/$LCODE/lang/silphones.csl data/$LCODE/lang/nonsilphones.csl
  echo "<UNK>" > data/$LCODE/lang/oov.txt

# (2) Create the L.fst without disambiguation symbols, for use in training.
  make_lexicon_fst.pl data/$LCODE/local/lexicon_${LCODE}.txt 0.5 SIL \
    | fstcompile --isymbols=data/$LCODE/lang/phones.txt \
      --osymbols=data/$LCODE/lang/words.txt --keep_isymbols=false \
      --keep_osymbols=false \
    | fstarcsort --sort_type=olabel > data/$LCODE/lang/L.fst

# (3) Create phonesets.txt and extra_questions.txt.
  gp_make_questions.pl -i data/$LCODE/lang/phones.txt \
    -m data/$LCODE/lang/phonesets_mono.txt -r data/$LCODE/lang/roots.txt
  # gp_extra_questions_${LCODE}.pl -i data/$LCODE/lang/phones.txt \
  #   -e data/$LCODE/lang/extra_questions.txt
  grep -v SIL data/$LCODE/lang/phonesets_mono.txt \
    > data/$LCODE/lang/phonesets_cluster.txt

# (4), Finally, for training, create the HMM topology prototype:
  silphonelist=`cat data/$LCODE/lang/silphones.csl | sed 's/:/ /g'`
  nonsilphonelist=`cat data/$LCODE/lang/nonsilphones.csl | sed 's/:/ /g'`
  sed -e "s:NONSILENCEPHONES:$nonsilphonelist:" \
    -e "s:SILENCEPHONES:$silphonelist:" $PROTO > data/$LCODE/lang/topo

done

echo "Preparing test data"

for LCODE in GE PO SP SW; do
# (0) Copy over some files common to traina and test:
  mkdir -p data/$LCODE/lang_test
  for f in phones.txt words.txt L.fst silphones.csl nonsilphones.csl; do
    cp data/$LCODE/lang/$f -t data/$LCODE/lang_test/
  done

# (1) Create a list of phones including the disambiguation symbols.
#     --include-zero includes the #0 symbol that is passed from G.fst
  ndisambig=`cat data/$LCODE/local/lex_ndisambig`;
  add_disambig.pl --include-zero data/$LCODE/lang_test/phones.txt $ndisambig \
    > data/$LCODE/lang_test/phones_disambig.txt
  cp data/$LCODE/lang_test/phones_disambig.txt -t data/$LCODE/lang/  # for MMI.

# (2) Create the lexicon FST with disambiguation symbols. There is an extra
#     step where we create a loop to "pass through" the disambiguation symbols
#     from G.fst.  
  phone_disambig_symbol=`grep \#0 data/$LCODE/lang_test/phones_disambig.txt | awk '{print $2}'`
  word_disambig_symbol=`grep \#0 data/$LCODE/lang_test/words.txt | awk '{print $2}'`

  make_lexicon_fst.pl data/$LCODE/local/lexicon_disambig_${LCODE}.txt 0.5 SIL \
    '#'$ndisambig \
    | fstcompile --isymbols=data/$LCODE/lang_test/phones_disambig.txt \
      --osymbols=data/$LCODE/lang_test/words.txt --keep_isymbols=false \
      --keep_osymbols=false \
    | fstaddselfloops  "echo $phone_disambig_symbol |" \
      "echo $word_disambig_symbol |" \
    | fstarcsort --sort_type=olabel > data/$LCODE/lang_test/L_disambig.fst

  # Needed for discriminative training
  cp data/$LCODE/lang_test/L_disambig.fst -t data/$LCODE/lang/

# (3) Create L_align.fst, which is as L.fst but with alignment symbols (#1 
#     and #2 at the beginning and end of words, on the input side). These are 
#     used to work out word boundaries. Useful if we ever need to create ctm's
  cat data/$LCODE/local/lexicon_${LCODE}.txt \
    | awk '{printf("%s #1 ", $1); for (n=2; n <= NF; n++) 
           { printf("%s ", $n); } print "#2"; }' \
    | make_lexicon_fst.pl - 0.5 SIL \
    | fstcompile --isymbols=data/$LCODE/lang_test/phones_disambig.txt \
      --osymbols=data/$LCODE/lang_test/words.txt --keep_isymbols=false \
      --keep_osymbols=false \
    | fstarcsort --sort_type=olabel > data/$LCODE/lang_test/L_align.fst

done

# Convert the different available language models to FSTs, and create separate 
# decoding configurations for each. -- This is very Edinburgh specific.

# TODO(arnab): The core formatting is done in a format_lm fucntion inside this 
# script, which will be common across setups, so it can probably be taken out 
# and put as a separate script in the utils directory.
gp_format_lms_edin.sh data

echo "Succeeded in formatting data."
