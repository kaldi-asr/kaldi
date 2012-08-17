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
set -o pipefail

. ./path.sh

# Begin configuration section.
filter_vocab_sri=false    # if true, use SRILM to change the LM vocab
# end configuration sections

help_message="Usage: "`basename $0`" [options] LC [LC ... ]
where LC is a 2-letter code for GlobalPhone languages.
options: 
  --help                             # print this message and exit
  --filter-vocab-sri (true|false)    # default: false; if true, use SRILM to change the LM vocab.
";

. utils/parse_options.sh

if [ $# -lt 1 ]; then
  printf "$help_message\n"; exit 1;
fi

LANGUAGES=
while [ $# -gt 0 ]; do
  case "$1" in
  ??) LANGUAGES=$LANGUAGES" $1"; shift ;;
  *)  echo "Unknown argument: $1, exiting"; error_exit $usage ;;
  esac
done

[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables

printf "Preparing train/test data ... "

for L in $LANGUAGES; do
# (0) Create a directory to contain files needed in training:
  for x in train dev eval; do 
    mkdir -p data/$L/$x
    cp data/$L/local/${x}_${L}_wav.scp data/$L/$x/wav.scp
    cp data/$L/local/${x}_${L}.txt data/$L/$x/text
    cp data/$L/local/${x}_${L}.spk2utt data/$L/$x/spk2utt
    cp data/$L/local/${x}_${L}.utt2spk data/$L/$x/utt2spk
  done
done

echo "Done"

for L in $LANGUAGES; do
  for lm_suffix in tg tgpr; do
    test=data/$L/lang_test_${lm_suffix}
    mkdir -p $test
    if $filter_vocab_sri; then  # use SRILM to change LM vocab
      utils/format_lm_sri.sh data/$L/lang data/$L/lm/${L}.${lm_suffix}.arpa.gz \
	data/$L/local/dict/lexicon.txt "${test}_sri"
    else  # just remove out-of-lexicon words without renormalizing the LM
      utils/format_lm.sh data/$L/lang data/$L/lm/${L}.${lm_suffix}.arpa.gz \
	data/$L/local/dict/lexicon.txt "$test"
    fi
  done
done
