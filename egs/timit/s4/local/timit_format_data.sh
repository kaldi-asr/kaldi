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
Prepare train, dev, test file lists.\n\n
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
. path.sh

echo "Preparing train data"

# (0) Create a directory to contain files needed in training:
for x in train dev test; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp
  cp data/local/${x}.trans2 data/$x/text
  cp data/local/${x}.spk2utt data/$x/spk2utt
  cp data/local/${x}.utt2spk data/$x/utt2spk
done

mkdir -p data/lang
cp data/local/phones.txt -t data/lang/
cp data/local/words.txt -t data/lang/

# (1) Generate colon-separated lists of silence and non-silence phones
silphones="cl epi sil vcl";
silphones.pl data/lang/phones.txt "$silphones" \
  data/lang/silphones.csl data/lang/nonsilphones.csl

# (2) Create the L.fst without disambiguation symbols, for use in training.
make_lexicon_fst.pl data/local/lexicon.txt 0.5 sil \
  | fstcompile --isymbols=data/lang/phones.txt \
    --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false \
  | fstarcsort --sort_type=olabel > data/lang/L.fst

# (3) Create phonesets.txt and extra_questions.txt.
timit_make_questions.pl -i data/lang/phones.txt \
  -m data/lang/phonesets_mono.txt -r data/lang/roots.txt
grep -v sil data/lang/phonesets_mono.txt \
  > data/lang/phonesets_cluster.txt
echo "cl epi sil vcl" > data/lang/extra_questions.txt

# (4), Finally, for training, create the HMM topology prototype:
silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
sed -e "s:NONSILENCEPHONES:$nonsilphonelist:" \
  -e "s:SILENCEPHONES:$silphonelist:" $PROTO > data/lang/topo

echo "Preparing test data"

# (0) Copy over some files common to traina and test:
mkdir -p data/lang_test
for f in phones.txt words.txt L.fst silphones.csl nonsilphones.csl; do
  cp data/lang/$f -t data/lang_test/
done

# (1) Create a list of phones including the disambiguation symbols.
#     --include-zero includes the #0 symbol that is passed from G.fst
ndisambig=`cat data/local/lex_ndisambig`;
add_disambig.pl --include-zero data/lang_test/phones.txt $ndisambig \
  > data/lang_test/phones_disambig.txt
cp data/lang_test/phones_disambig.txt -t data/lang/  # for MMI.

# (2) Create the lexicon FST with disambiguation symbols. There is an extra
#     step where we create a loop to "pass through" the disambiguation symbols
#     from G.fst.  
phone_disambig_symbol=`grep \#0 data/lang_test/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/lang_test/words.txt | awk '{print $2}'`

make_lexicon_fst.pl data/local/lexicon_disambig.txt 0.5 sil '#'$ndisambig \
  | fstcompile --isymbols=data/lang_test/phones_disambig.txt \
    --osymbols=data/lang_test/words.txt --keep_isymbols=false \
    --keep_osymbols=false \
  | fstaddselfloops  "echo $phone_disambig_symbol |" \
    "echo $word_disambig_symbol |" \
  | fstarcsort --sort_type=olabel > data/lang_test/L_disambig.fst

  # Needed for discriminative training
cp data/lang_test/L_disambig.fst -t data/lang/

# (3) Convert the language model to FST, and create decoding configuration.
timit_format_lms.sh data

echo "Succeeded in formatting data."
