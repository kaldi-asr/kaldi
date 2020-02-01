#!/usr/bin/env bash

# Copyright 2013  Arnab Ghoshal
#                 Johns Hopkins University (author: Daniel Povey)
#           2014  Guoguo Chen

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
weblm=
# end configuration sections

help_message="Usage: $0 [options] <train-txt> <dict> <out-dir> [fisher-dirs]
Train language models for Switchboard-1, and optionally for Fisher and \n
web-data from University of Washington.\n
options: 
  --help          # print this message and exit
  --weblm DIR     # directory for web-data from University of Washington
";

. utils/parse_options.sh

if [ $# -lt 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

text=$1     # data/local/train/text
lexicon=$2  # data/local/dict/lexicon.txt
dir=$3      # data/local/lm

shift 3
fisher_dirs=( $@ )

for f in "$text" "$lexicon"; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
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
cut -d' ' -f2- $text | tail -n +$(($heldout_sent+1)) | gzip -c > $dir/train.gz
cut -d' ' -f2- $text | head -n $heldout_sent > $dir/heldout

cut -d' ' -f1 $lexicon > $dir/wordlist

# Trigram language model
ngram-count -text $dir/train.gz -order 3 -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/sw1.o3g.kn.gz
echo "PPL for SWBD1 trigram LM:"
ngram -unk -lm $dir/sw1.o3g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/sw1.o3g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/3gram.ppl2
# file data/local/lm/heldout: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -250952 ppl= 90.5071 ppl1= 132.479

# 4gram language model
ngram-count -text $dir/train.gz -order 4 -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/sw1.o4g.kn.gz
echo "PPL for SWBD1 4gram LM:"
ngram -unk -lm $dir/sw1.o4g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/sw1.o4g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/4gram.ppl2
# file data/local/lm/heldout: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -253747 ppl= 95.1632 ppl1= 139.887

mkdir -p $dir/fisher
rm -rf $dir/fisher/text0
for x in ${fisher_dirs[@]}; do
  [ ! -d $x/data/trans ] \
    && "$0: Cannot find transcripts in Fisher directory $x" && exit 1;
  cat $x/data/trans/*/*.txt |\
    grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $dir/fisher/text0
done

if [ -f $dir/fisher/text0 ]; then
  cat $dir/fisher/text0 | local/fisher_map_words.pl \
    | gzip -c > $dir/fisher/text1.gz

  for x in 3 4; do
    ngram-count -text $dir/fisher/text1.gz -order $x -limit-vocab \
      -vocab $dir/wordlist -unk -map-unk "<unk>" -kndiscount -interpolate \
      -lm $dir/fisher/fisher.o${x}g.kn.gz
    echo "PPL for Fisher ${x}gram LM:"
    ngram -unk -lm $dir/fisher/fisher.o${x}g.kn.gz -ppl $dir/heldout
    ngram -unk -lm $dir/fisher/fisher.o${x}g.kn.gz -ppl $dir/heldout -debug 2 \
      >& $dir/fisher/${x}gram.ppl2
    compute-best-mix $dir/${x}gram.ppl2 \
      $dir/fisher/${x}gram.ppl2 >& $dir/sw1_fsh_mix.${x}gram.log
    grep 'best lambda' $dir/sw1_fsh_mix.${x}gram.log | perl -e '
      $_=<>;
      s/.*\(//; s/\).*//;
      @A = split;
      die "Expecting 2 numbers; found: $_" if(@A!=2);
      print "$A[0]\n$A[1]\n";' > $dir/sw1_fsh_mix.${x}gram.weights
    swb1_weight=$(head -1 $dir/sw1_fsh_mix.${x}gram.weights)
    fisher_weight=$(tail -n 1 $dir/sw1_fsh_mix.${x}gram.weights)
    ngram -order $x -lm $dir/sw1.o${x}g.kn.gz -lambda $swb1_weight \
      -mix-lm $dir/fisher/fisher.o${x}g.kn.gz \
      -unk -write-lm $dir/sw1_fsh.o${x}g.kn.gz
    echo "PPL for SWBD1 + Fisher ${x}gram LM:"
    ngram -unk -lm $dir/sw1_fsh.o${x}g.kn.gz -ppl $dir/heldout
  done
fi

if [ ! -z "$weblm" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The following takes about 11 minutes to download on Eddie: 
# wget --no-check-certificate http://ssli.ee.washington.edu/data/191M_conversational_web-filt+periods.gz

