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

. ./path.sh    # Sets the PATH to contain necessary executables

# Begin configuration section.
filter_vocab_sri=false    # if true, use SRILM to change the LM vocab
srilm_opts="-subset -prune-lowprobs -unk -tolower"
# end configuration sections

help_message="Usage: "`basename $0`" [options] LM-dir LC [LC ... ]
where LC is a 2-letter code for GlobalPhone languages, and LM-dir is assumed to 
contain LMs for all the languages (e.g. RU.3gram.lm.gz for Russian).
options: 
  --help                           # print this message and exit
  --filter-vocab-sri (true|false)  # use SRILM to change the LM vocab (default: $filter_vocab_sri)
  --srilm-opts STRING              # options to pass to SRILM tools (default: '$srilm_opts')
";

. utils/parse_options.sh

if [ $# -lt 2 ]; then
  printf "$help_message\n"; exit 1;
fi

LMDIR=$1; shift;
LANGUAGES=
while [ $# -gt 0 ]; do
  case "$1" in
  ??) LANGUAGES=$LANGUAGES" $1"; shift ;;
  *)  echo "Unknown argument: $1, exiting"; error_exit $usage ;;
  esac
done

for L in $LANGUAGES; do
  lm=$LMDIR/${L}.3gram.lm.gz
  [ -f $lm ] || { echo "LM '$lm' not found"; exit 1; }
  test=data/$L/lang_test_tg
  if $filter_vocab_sri; then  # use SRILM to change LM vocab
    utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
      data/$L/lang $lm data/$L/local/dict/lexicon.txt "${test}_sri"
  else  # just remove out-of-lexicon words without renormalizing the LM
    utils/format_lm.sh data/$L/lang $lm data/$L/local/dict/lexicon.txt "$test"
  fi

  # Create a pruned version of the LM for building the decoding graphs, using 
  # 'prune-lm' from IRSTLM:
  mkdir -p data/$L/local/lm
  prune-lm --threshold=1e-7 $lm /dev/stdout | gzip -c \
    > data/$L/local/lm/${L}.tgpr.arpa.gz
  lm=data/$L/local/lm/${L}.tgpr.arpa.gz
  test=data/$L/lang_test_tgpr
  if $filter_vocab_sri; then  # use SRILM to change LM vocab
    utils/format_lm_sri.sh data/$L/lang $lm data/$L/local/dict/lexicon.txt \
      "${test}_sri"
  else  # just remove out-of-lexicon words without renormalizing the LM
    utils/format_lm.sh data/$L/lang $lm data/$L/local/dict/lexicon.txt "$test"
  fi
done
