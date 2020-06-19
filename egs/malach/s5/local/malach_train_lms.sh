#!/usr/bin/env bash

# Copyright 2019, IBM Research (Author: Michael Picheny) Adapted AMI recipe to MALACH Corpus
# Copyright 2013  Arnab Ghoshal, Pawel Swietojanski

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

# To be run from one directory above this script.

# Begin configuration section.
fisher=
order=4
swbd=
google=
web_sw=
web_fsh=
web_mtg=
# end configuration sections

help_message="Usage: "`basename $0`" [options] <train-txt> <dev-txt> <dict> <out-dir>
Train language models for Malach \n
options:
  --help          # print this message and exit
  --order N       # N-gram order (default: '$order')
";

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

train=$1    # data/train/text
dev=$2      # data/dev/text
lexicon=$3  # data/dict/lexicon.txt
dir=$4      # data/local/lm

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

set -o errexit
mkdir -p $dir
export LC_ALL=C

if ! command -v ngram-count 2>/dev/null; then
  echo "$0: SRILM is not installed.  Please install SRILM with:"
  echo "pushd $KALDI_ROOT; cd tools; extras/install_srilm.sh; popd"
  echo "[note: this may require registering on the SRILM website.]"
  exit 1
fi

cut -d' ' -f6- $train | gzip -c > $dir/train.gz
cut -d' ' -f6- $dev | gzip -c > $dir/dev.gz

awk '{print $1}' $lexicon | sort -u > $dir/wordlist.lex
gunzip -c $dir/train.gz | tr ' ' '\n' | grep -v ^$ | sort -u > $dir/wordlist.train
sort -u $dir/wordlist.lex $dir/wordlist.train > $dir/wordlist

ngram-count -text $dir/train.gz -order $order -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/malach.o${order}g.kn.gz \
  -gt3min 1 -gt4min 1
echo "PPL for MALACH LM:"
ngram -unk -lm $dir/malach.o${order}g.kn.gz -ppl $dir/dev.gz
ngram -unk -lm $dir/malach.o${order}g.kn.gz -ppl $dir/dev.gz -debug 2 >& $dir/ppl2
mix_ppl="$dir/ppl2"
mix_tag="malach"
mix_lms=( "$dir/malach.o${order}g.kn.gz" )
num_lms=1

#save the lm name for further use
echo "${mix_tag}.o${order}g.kn" > $dir/final_lm

