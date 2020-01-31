#!/usr/bin/env bash

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
order=3
swbd=
google=
web_sw=
web_fsh=
web_mtg=
# end configuration sections

help_message="Usage: "`basename $0`" [options] <train-txt> <dev-txt> <dict> <out-dir>
Train language models for AMI and optionally for Switchboard, Fisher and web-data from University of Washington.\n
options:
  --help          # print this message and exit
  --fisher DIR    # directory for Fisher transcripts
  --order N       # N-gram order (default: '$order')
  --swbd DIR      # Directory for Switchboard transcripts
  --web-sw FILE   # University of Washington (191M) Switchboard web data
  --web-fsh FILE  # University of Washington (525M) Fisher web data
  --web-mtg FILE  # University of Washington (150M) CMU+ICSI+NIST meeting data
";

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

train=$1    # data/ihm/train/text
dev=$2      # data/ihm/dev/text
lexicon=$3  # data/ihm/dict/lexicon.txt
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
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm $dir/ami.o${order}g.kn.gz
echo "PPL for AMI LM:"
ngram -unk -lm $dir/ami.o${order}g.kn.gz -ppl $dir/dev.gz
ngram -unk -lm $dir/ami.o${order}g.kn.gz -ppl $dir/dev.gz -debug 2 >& $dir/ppl2
mix_ppl="$dir/ppl2"
mix_tag="ami"
mix_lms=( "$dir/ami.o${order}g.kn.gz" )
num_lms=1

if [ ! -z "$swbd" ]; then
  mkdir -p $dir/swbd

  find $swbd -iname '*-trans.text' -exec cat {} \; | cut -d' ' -f4- \
    | gzip -c > $dir/swbd/text0.gz
  gunzip -c $dir/swbd/text0.gz | swbd_map_words.pl | gzip -c \
    > $dir/swbd/text1.gz
  ngram-count -text $dir/swbd/text1.gz -order $order -limit-vocab \
    -vocab $dir/wordlist -unk -map-unk "<unk>" -kndiscount -interpolate \
    -lm $dir/swbd/swbd.o${order}g.kn.gz
  echo "PPL for SWBD LM:"
  ngram -unk -lm $dir/swbd/swbd.o${order}g.kn.gz -ppl $dir/dev.gz
  ngram -unk -lm $dir/swbd/swbd.o${order}g.kn.gz -ppl $dir/dev.gz -debug 2 \
    >& $dir/swbd/ppl2

  mix_ppl="$mix_ppl $dir/swbd/ppl2"
  mix_tag="${mix_tag}_swbd"
  mix_lms=("${mix_lms[@]}" "$dir/swbd/swbd.o${order}g.kn.gz")
  num_lms=$[ num_lms + 1 ]
fi

if [ ! -z "$fisher" ]; then
  [ ! -d "$fisher/data/trans" ] \
    && echo "Cannot find transcripts in Fisher directory: '$fisher'" \
    && exit 1;
  mkdir -p $dir/fisher

  find $fisher -follow -path '*/trans/*fe*.txt' -exec cat {} \; | grep -v ^# | grep -v ^$ \
    | cut -d' ' -f4- | gzip -c > $dir/fisher/text0.gz
  gunzip -c $dir/fisher/text0.gz | local/fisher_map_words.pl \
    | gzip -c > $dir/fisher/text1.gz
  ngram-count -debug 0 -text $dir/fisher/text1.gz -order $order -limit-vocab \
    -vocab $dir/wordlist -unk -map-unk "<unk>" -kndiscount -interpolate \
    -lm $dir/fisher/fisher.o${order}g.kn.gz
  echo "PPL for Fisher LM:"
  ngram -unk -lm $dir/fisher/fisher.o${order}g.kn.gz -ppl $dir/dev.gz
  ngram -unk -lm $dir/fisher/fisher.o${order}g.kn.gz -ppl $dir/dev.gz -debug 2 \
   >& $dir/fisher/ppl2

  mix_ppl="$mix_ppl $dir/fisher/ppl2"
  mix_tag="${mix_tag}_fsh"
  mix_lms=("${mix_lms[@]}" "$dir/fisher/fisher.o${order}g.kn.gz")
  num_lms=$[ num_lms + 1 ]
fi

if [ ! -z "$google1B" ]; then
  mkdir -p $dir/google
  wget -O $dir/google/cantab.lm3.bz2 http://vm.cantabresearch.com:6080/demo/cantab.lm3.bz2
  wget -O $dir/google/150000.lex http://vm.cantabresearch.com:6080/demo/150000.lex

  ngram -unk -limit-vocab -vocab $dir/wordlist -lm $dir/google.cantab.lm3.bz3 \
     -write-lm $dir/google/google.o${order}g.kn.gz

  mix_ppl="$mix_ppl $dir/goog1e/ppl2"
  mix_tag="${mix_tag}_fsh"
  mix_lms=("${mix_lms[@]}" "$dir/google/google.o${order}g.kn.gz")
  num_lms=$[ num_lms + 1 ]
fi

## The University of Washington conversational web data can be obtained as:
## wget --no-check-certificate http://ssli.ee.washington.edu/data/191M_conversational_web-filt+periods.gz
if [ ! -z "$web_sw" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The University of Washington Fisher conversational web data can be obtained as:
## wget --no-check-certificate http://ssli.ee.washington.edu/data/525M_fisher_conv_web-filt+periods.gz
if [ ! -z "$web_fsh" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The University of Washington meeting web data can be obtained as:
## wget --no-check-certificate http://ssli.ee.washington.edu/data/150M_cmu+icsi+nist-meetings.gz
if [ ! -z "$web_mtg" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

if [ $num_lms -gt 1  ]; then
  echo "Computing interpolation weights from: $mix_ppl"
  compute-best-mix $mix_ppl >& $dir/mix.log
  grep 'best lambda' $dir/mix.log \
    | perl -e '$_=<>; s/.*\(//; s/\).*//; @A = split; for $i (@A) {print "$i\n";}' \
    > $dir/mix.weights
  weights=( `cat $dir/mix.weights` )
  cmd="ngram -lm ${mix_lms[0]} -lambda 0.715759 -mix-lm ${mix_lms[1]}"
  for i in `seq 2 $((num_lms-1))`; do
    cmd="$cmd -mix-lm${i} ${mix_lms[$i]} -mix-lambda${i} ${weights[$i]}"
  done
  cmd="$cmd -unk -write-lm $dir/${mix_tag}.o${order}g.kn.gz"
  echo "Interpolating LMs with command: \"$cmd\""
  $cmd
  echo "PPL for the interolated LM:"
  ngram -unk -lm $dir/${mix_tag}.o${order}g.kn.gz -ppl $dir/dev.gz
fi

#save the lm name for further use
echo "${mix_tag}.o${order}g.kn" > $dir/final_lm

