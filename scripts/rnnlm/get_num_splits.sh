#!/bin/bash

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.


# This script works out how many pieces we want to split the data into for a
# particular training run, based on how many words are in the data directory
# (excluding dev), and the target words-per-split.


if [ $# != 2 ]; then
  (
    echo "Usage: rnnlm/get_num_splits.sh <target-words-per-split> <data-dir>"
    echo "e.g.: rnnlm/get_num_splits.sh 200000 data/text"
    echo "This works out how many pieces to split a data directory into; it"
    echo "echoes a number such that the average words per split does not exceed"
    echo "<target-words-per-split>.  It works out the number of words of training data from"
    echo "<data-dir>/*.counts."
  ) 1>&2
  exit 1
fi


words_per_split=$1
data=$2

! [ $words_per_split -eq $words_per_split ] && \
  echo "$0: first arg must be an integer" 1>&2 && exit 1;

[ ! -d $data ] && \
  echo "$0: no such directory $data" 1>&2 && exit 1;

set -e -o pipefail -u

export LC_ALL=C


tot=0

for f in $data/*.counts; do
  if [ "$f" != "$data/dev.counts" ]; then
    this_tot=$(cat $f | awk '{tot += $2} END{print tot}')
    if ! [ $this_tot -gt 0 ]; then
      echo "$0: there were no counts in counts file $f" 1>&2
      exit 1
    fi
    tot=$[tot+this_tot]
  fi
done

if ! [ $tot -gt 0 ]; then
  echo "$0: there was a problem getting counts from directory $data (no counts present?)" 1>&2
  exit 1
fi

# adding words_per_split-1 below causes us to round up the number of splits.
num_splits=$[(tot+words_per_split-1)/words_per_split]

if ! [ $num_splits -gt 0 ]; then
  echo "$0: there was a problem getting the number of splits" 1>&2
  exit 1
fi


echo "get_num_splits.sh: based on num-words=$tot and target-words-per-split=$words_per_split, got $num_splits splits." 1>&2

echo $num_splits  # this is the only thing that goes to the standard output.




