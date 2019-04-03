#!/bin/bash

# Copyright 2017  Intellisist, Inc. (Author: Navneeth K)
#           2017  Xiaohui Zhang
#           2018  Ruizhe Huang
# Apache License 2.0

# This script trains a g2p model using Phonetisaurus.

stage=0
encoding='utf-8'
only_words=true
silence_phones=

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <lexicon-in> <work-dir>"
  echo "    where <lexicon-in> is the training lexicon (one pronunciation per "
  echo "    word per line, with lines like 'hello h uh l ow') and"
  echo "    <work-dir> is directory where the models will be stored"
  echo "e.g.: $0 --silence-phones data/local/dict/silence_phones.txt data/local/dict/lexicon.txt exp/g2p/"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --silence-phones <silphones-list>                # e.g. data/local/dict/silence_phones.txt."
  echo "                                                   # A list of silence phones, one or more per line"
  echo "                                                   # Relates to  --only-words option"
  echo "  --only-words (true|false)    (default: true)     # If true, exclude silence words, i.e."
  echo "                                                   # words with one or multiple phones which are all silence."
  exit 1;
fi

lexicon=$1
wdir=$2

[ ! -f $lexicon ] && echo "Cannot find $lexicon" && exit

isuconv=`which uconv`
if [ -z $isuconv ]; then
  echo "uconv was not found. You must install the icu4c package."
  exit 1;
fi

if ! phonetisaurus=`which phonetisaurus-apply` ; then
  echo "Phonetisarus was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_phonetisaurus.sh"
  exit 1
fi

mkdir -p $wdir


# For input lexicon, remove pronunciations containing non-utf-8-encodable characters,
# and optionally remove words that are mapped to a single silence phone from the lexicon.
if [ $stage -le 0 ]; then
  if $only_words && [ ! -z "$silence_phones" ]; then
    awk 'NR==FNR{a[$1] = 1; next} {s=$2;for(i=3;i<=NF;i++) s=s" "$i; if(!(s in a)) print $1" "s}' \
      $silence_phones $lexicon | \
      awk '{printf("%s\t",$1); for (i=2;i<NF;i++){printf("%s ",$i);} printf("%s\n",$NF);}' | \
      uconv -f "$encoding"  -t "$encoding" -x Any-NFC - | awk 'NF > 0'> $wdir/lexicon_tab_separated.txt
  else
    awk '{printf("%s\t",$1); for (i=2;i<NF;i++){printf("%s ",$i);} printf("%s\n",$NF);}' $lexicon | \
      uconv -f "$encoding" -t "$encoding" -x Any-NFC - | awk 'NF > 0'> $wdir/lexicon_tab_separated.txt
  fi
fi

if [ $stage -le 1 ]; then
  # Align lexicon stage. Lexicon is assumed to have first column tab separated
  phonetisaurus-align --input=$wdir/lexicon_tab_separated.txt --ofile=${wdir}/aligned_lexicon.corpus || exit 1;
fi

if [ $stage -le 2 ]; then
  # Convert aligned lexicon to arpa using make_kn_lm.py, a re-implementation of srilm's ngram-count functionality.
  ./utils/lang/make_kn_lm.py -ngram-order 7 -text ${wdir}/aligned_lexicon.corpus -lm ${wdir}/aligned_lexicon.arpa
fi

if [ $stage -le 3 ]; then
  # Convert the arpa file to FST.
  phonetisaurus-arpa2wfst --lm=${wdir}/aligned_lexicon.arpa --ofile=${wdir}/model.fst
fi

