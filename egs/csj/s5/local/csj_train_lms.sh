#!/bin/bash 

# 2016 Modified by Takafumi Moriya at Tokyo Institute of Technology
# for Japanese speech recognition using CSJ.

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
weblm=
# end configuration sections

help_message="Usage: "`basename $0`" [options] <train-txt> <dict> <out-dir>
Train language models for CSJ.\n
options: 
  --help          # print this message and exit
";

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

text=$1    
lexicon=$2 
dir=$3     

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
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/csj.o3g.kn.gz
echo "PPL for CSJ LM:"
ngram -unk -lm $dir/csj.o3g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/csj.o3g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/ppl2
# file data/local/lm/heldout: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -250952 ppl= 90.5071 ppl1= 132.479


if [ ! -z "$weblm" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The following takes about 11 minutes to download on Eddie: 
# wget --no-check-certificate http://ssli.ee.washington.edu/data/191M_conversational_web-filt+periods.gz

