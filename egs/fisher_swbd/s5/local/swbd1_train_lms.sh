#!/bin/bash -v

# Copyright 2013  Arnab Ghoshal
#                 Johns Hopkins University (author: Daniel Povey)

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
weblm=
# end configuration sections

help_message="Usage: "`basename $0`" [options] <train-txt> <dict> <out-dir>
Train language models for Switchboard-1, and optionally for Fisher and web-data from University of Washington.\n
options: 
  --help          # print this message and exit
  --fisher DIR    # directory for Fisher transcripts (default: '$srilm_opts')
  --weblm DIR     # directory for web-data from University of Washington
";

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

text=$1     # data/local/train/text
lexicon=$2  # data/local/dict/lexicon.txt
dir=$3      # data/local/lm

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64 
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi
    

set -o errexit
mkdir -p $dir
export LC_ALL=C 

heldout_sent=10000
cut -d' ' -f2- $text | gzip -c > $dir/train.all.gz
cut -d' ' -f2- $text | tail -n +$heldout_sent | gzip -c > $dir/train.gz
cut -d' ' -f2- $text | head -n $heldout_sent > $dir/heldout

cut -d' ' -f1 $lexicon > $dir/wordlist

ngram-count -text $dir/train.gz -order 3 -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/sw1.o3g.kn.gz
echo "PPL for SWBD1 LM:"
ngram -unk -lm $dir/sw1.o3g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/sw1.o3g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/ppl2
# file data/local/lm/heldout: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -250952 ppl= 90.5071 ppl1= 132.479


if [ ! -z "$fisher" ]; then
  [ ! -d "$fisher/data/trans" ] \
    && echo "Cannot find transcripts in Fisher directory: '$fisher'" \
    && exit 1;
  mkdir -p $dir/fisher

  cat $fisher/data/trans/*/*.TXT | grep -v ^# | grep -v ^$ | cut -d' ' -f4- \
    | gzip -c > $dir/fisher/text0.gz
  gunzip -c $dir/fisher/text0.gz | local/fisher_map_words.pl \
    | gzip -c > $dir/fisher/text1.gz
  ngram-count -text $dir/fisher/text1.gz -order 3 -limit-vocab \
    -vocab $dir/wordlist -unk -map-unk "<unk>" -kndiscount -interpolate \
    -lm $dir/fisher/fisher.o3g.kn.gz
  echo "PPL for Fisher LM:"
  ngram -unk -lm $dir/fisher/fisher.o3g.kn.gz -ppl $dir/heldout
  ngram -unk -lm $dir/fisher/fisher.o3g.kn.gz -ppl $dir/heldout -debug 2 \
    >& $dir/fisher/ppl2
  compute-best-mix $dir/ppl2 $dir/fisher/ppl2 >& $dir/sw1_fsh_mix.log
  grep 'best lambda' $dir/sw1_fsh_mix.log \
    | perl -e '$_=<>; s/.*\(//; s/\).*//; @A = split; die "Expecting 2 numbers; found: $_" if(@A!=2); print "$A[0]\n$A[1]\n";' > $dir/sw1_fsh_mix.weights
  swb1_weight=$(head -1 $dir/sw1_fsh_mix.weights)
  fisher_weight=$(tail -n 1 $dir/sw1_fsh_mix.weights)
  ngram -lm $dir/sw1.o3g.kn.gz -lambda $swb1_weight \
    -mix-lm $dir/fisher/fisher.o3g.kn.gz \
    -unk -write-lm $dir/sw1_fsh.o3g.kn.gz
  echo "PPL for SWBD1 + Fisher LM:"
  ngram -unk -lm $dir/sw1_fsh.o3g.kn.gz -ppl $dir/heldout
fi

if [ ! -z "$weblm" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The following takes about 11 minutes to download on Eddie: 
# wget --no-check-certificate http://ssli.ee.washington.edu/data/191M_conversational_web-filt+periods.gz

