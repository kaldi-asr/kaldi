#!/bin/bash
# Copyright (c) 2015-2018, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
silence_word=
filter='OOV=0'
# End configuration section
echo $0 "$@"
. ./utils/parse_options.sh || exit 1;

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


data=$1
lang=$2
workdir=$3

mkdir -p $workdir
if [ -f $data/categories ] ; then
  cat $data/categories | \
    local/search/filter_by_category.pl $data/categories "$filter" > $workdir/categories

  if [ ! -s $workdir/categories ]; then
    echo "$0: WARNING: $workdir/categories is zero-size. That means no keyword"
    echo "$0: WARNING: was found that fits the filter \"$filter\". That might be expected."
    touch $workdir/keywords.int
    touch $workdir/keywords.fsts
    exit 0
  fi
  grep -w -F -f <(awk '{print $1}' $workdir/categories) \
    $data/keywords.int > $workdir/keywords.int
else
  cp $data/keywords.int $workdir/keywords.int
fi



if [ -s $workdir/keywords.int ]; then
  if [ -z $silence_word ]; then
    transcripts-to-fsts ark:$workdir/keywords.int \
      ark,scp,t:$workdir/keywords.fsts,- | sort -o $workdir/keywords.scp
  else
    silence_int=`grep -w $silence_word $lang/words.txt | awk '{print $2}'`
    [ -z $silence_int ] && \
       echo "$0: Error: could not find integer representation of silence word $silence_word" && exit 1;
    transcripts-to-fsts ark:$data/keywords.int ark,t:- | \
      awk -v 'OFS=\t' -v silint=$silence_int '{
        if (NF == 4 && $1 != 0) { print $1, $1, silint, silint; } print;
      }' | fstcopy ark:- ark,scp,t:$workdir/keywords.fsts,- | \
      sort -o $workdir/keywords.scp
  fi
else
  echo "$0: WARNING: $workdir/keywords.int is zero-size. That means no keyword"
  echo "$0: WARNING: was found in the dictionary. That might be expected -- or not."
  touch $workdir/keywords.fsts
fi

