#!/usr/bin/env bash

# Copyright 2013  Arnab Ghoshal
#                 Johns Hopkins University (author: Daniel Povey)
#           2014  Guoguo Chen
#           2019  Dongji Gao

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

help_message="Usage: $0 [options] <train-txt> <dict> <out-dir> [giga-dirs]
Train language models for GALE Arabic, and optionally for Gigaword.\n
options: 
  --help          # print this message and exit
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
giga_dirs=( $@ )

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
    
stage=0

set -o errexit
mkdir -p $dir
export LC_ALL=C 

heldout_sent=10000
cut -d' ' -f2- $text | gzip -c > $dir/train.all.gz
cut -d' ' -f2- $text | tail -n +$heldout_sent | gzip -c > $dir/train.gz
cut -d' ' -f2- $text | head -n $heldout_sent > $dir/heldout

cut -d' ' -f1 $lexicon > $dir/wordlist

if [ $stage -le 1 ]; then
  # Trigram language model
  echo "training tri-gram lm"
  smoothing="kn"
  ngram-count -text $dir/train.gz -order 3 -limit-vocab -vocab $dir/wordlist \
    -unk -map-unk "<UNK>" -${smoothing}discount -interpolate -lm $dir/gale.o3g.${smoothing}.gz
  echo "PPL for GALE Arabic trigram LM:"
  ngram -unk -lm $dir/gale.o3g.${smoothing}.gz -ppl $dir/heldout
  ngram -unk -lm $dir/gale.o3g.${smoothing}.gz -ppl $dir/heldout -debug 2 >& $dir/3gram.${smoothing}.ppl2
  
  # 4gram language model
  echo "training 4-gram lm"
  ngram-count -text $dir/train.gz -order 4 -limit-vocab -vocab $dir/wordlist \
    -unk -map-unk "<UNK>" -${smoothing}discount -interpolate -lm $dir/gale.o4g.${smoothing}.gz
  echo "PPL for GALE Arabic 4gram LM:"
  ngram -unk -lm $dir/gale.o4g.${smoothing}.gz -ppl $dir/heldout
  ngram -unk -lm $dir/gale.o4g.${smoothing}.gz -ppl $dir/heldout -debug 2 >& $dir/4gram.${smoothing}.ppl2
fi

if [ ! -z $giga_dirs ]; then
  mkdir -p $dir/giga
  if [ ! -f $giga_dirs/text.2000k ]; then
    echo "Arabic Gigaword text not found, prepare it"
    local/prepare_giga.sh $giga_dirs
  fi

  cp $giga_dirs/text.2000k $dir/giga
  cat $dir/giga/text.2000k | gzip -c > $dir/giga/text2000k.gz
  
  for x in 3 4; do
    smoothing="kn"
    ngram-count -text $dir/giga/text2000k.gz -order $x -limit-vocab \
      -vocab $dir/wordlist -unk -map-unk "<UNK>" -${smoothing}discount -interpolate \
      -lm $dir/giga/giga.o${x}g.${smoothing}.gz
    echo "PPL for Gigaword ${x}gram LM:"
    ngram -unk -lm $dir/giga/giga.o${x}g.${smoothing}.gz -ppl $dir/heldout
    ngram -unk -lm $dir/giga/giga.o${x}g.${smoothing}.gz -ppl $dir/heldout -debug 2 \
      >& $dir/giga/${x}gram.${smoothing}.ppl2
    compute-best-mix $dir/${x}gram.${smoothing}.ppl2 \
      $dir/giga/${x}gram.${smoothing}.ppl2 >& $dir/gale_giga_mix.${x}gram.${smoothing}.log
    grep 'best lambda' $dir/gale_giga_mix.${x}gram.${smoothing}.log | perl -e '
      $_=<>;
      s/.*\(//; s/\).*//;
      @A = split;
      die "Expecting 2 numbers; found: $_" if(@A!=2);
      print "$A[0]\n$A[1]\n";' > $dir/gale_giga_mix.${x}gram.${smoothing}.weights
    gale_weight=$(head -1 $dir/gale_giga_mix.${x}gram.${smoothing}.weights)
    giga_weight=$(tail -n 1 $dir/gale_giga_mix.${x}gram.${smoothing}.weights)
    ngram -order $x -lm $dir/gale.o${x}g.${smoothing}.gz -lambda $swb1_weight \
      -mix-lm $dir/giga/giga.o${x}g.${smoothing}.gz \
      -unk -write-lm $dir/gale_giga.o${x}g.${smoothing}.gz
    echo "PPL for GALE + Gigaword ${x}gram LM:"
    ngram -unk -lm $dir/gale_giga.o${x}g.${smoothing}.gz -ppl $dir/heldout
  done
fi
