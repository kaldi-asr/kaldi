#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

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
Prepare train, dev, test file lists for TIMIT.\n\n
Required arguments:\n
  --config-dir=DIR\tDirecory containing the necessary config files\n
  --corpus-dir=DIR\tDirectory for the GlobalPhone corpus\n
  --work-dir=DIR\t\tWorking directory\n
";

if [ $# -lt 3 ]; then
  error_exit $usage;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --config-dir=*)
  CONFDIR=`read_dirname $1`; shift ;;
  --corpus-dir=*)
  CORPUS=`read_dirname $1`; shift ;;
  --work-dir=*)
  WDIR=`read_dirname $1`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

# (1) check if the config files are in place:
cd $CONFDIR
[ -f test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";

cd $WDIR
[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables

# (2) get the various file lists (for audio, transcription, etc.)
mkdir -p data/local
timit_prep_flists.sh --corpus-dir=$CORPUS --dev-spk=$CONFDIR/dev_spk.list \
  --test-spk=$CONFDIR/test_spk.list --work-dir=data

# (3) Normalize the transcripts.
timit_norm_trans.pl -i data/local/train.trans -m $CONFDIR/phones.60-48-39.map \
  -to 48 > data/local/train.trans2;
for x in dev test; do
  timit_norm_trans.pl -i data/local/${x}.trans -m $CONFDIR/phones.60-48-39.map \
    -to 39 > data/local/${x}.trans2;
done

# Create the lexicon, which is just an identity mapping
cut -d' ' -f2- data/local/train.trans2 | tr ' ' '\n' | sort -u > data/local/p
paste data/local/p data/local/p > data/local/lexicon.txt

# add disambig symbols to the lexicon: TODO: delete
ndisambig=`add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1];  # add one disambig symbol for silence
echo $ndisambig > data/local/lex_ndisambig

# Get the list of phones and map them to integers (adding the null symbol <eps>
# to the list).
cut -f2 data/local/lexicon.txt \
  | awk 'BEGIN{ print "<eps> 0"; } { printf("%s %d\n", $1, NR); }' \
  > data/local/phones.txt

# Get the list of words:
cut -f1 data/local/lexicon.txt \
  | awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} 
         END{printf("#0 %d\n", NR+1);}' > data/local/words.txt

# (4) Create the phone bigram LM
(
  [ -z "$IRSTLM" ] && \
    error_exit "LM building wo'nt work without setting the IRSTLM env variable"
  cut -d' ' -f2- data/local/train.trans2 | sed -e 's:^:<s> :' -e 's:$: </s>:' \
    > data/local/lm_train.txt
  build-lm.sh -i data/local/lm_train.txt -n 2 -o data/local/lm_phone_bg.ilm.gz
  compile-lm data/local/lm_phone_bg.ilm.gz --text yes /dev/stdout \
    | grep -v unk | gzip -c > data/local/lm_phone_bg.arpa.gz 

) >& data/prepare_lm.log

echo "Finished data preparation."
