#!/bin/bash

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.


# This script works out how many pieces we want to split the data into for a
# particular training run, based on how many words are in the data directory
# (excluding dev), and the target words-per-split.


if [ $# != 3 ]; then
  (
    echo "Usage: rnnlm/get_num_splits.sh <target-words-per-split> <data-dir> <weights-file>"
    echo "e.g.: rnnlm/get_num_splits.sh 200000 data/text exp/rnnlm/data_weights.txt"
    echo "This works out how many pieces to split a data directory into; it"
    echo "echoes a number such that the average words per split does not exceed"
    echo "<target-words-per-split>.  It works out the number of words of training data from"
    echo "<data-dir>/*.counts; they are scaled by the data-multiplicities given as"
    echo "the second field of <weights-file> for each data source."
  ) 1>&2
  exit 1
fi


words_per_split=$1
data=$2
weights_file=$3

! [ $words_per_split -eq $words_per_split ] && \
  echo "$0: first arg must be an integer" 1>&2 && exit 1;

[ ! -d $data ] && \
  echo "$0: no such directory $data" 1>&2 && exit 1;

[ ! -f $weight ] && \
  echo "$0: expected weights file in $weight" 1>&2 && exit 1;

set -e -o pipefail -u

export LC_ALL=C





multiplicities=$(mktemp tmp.XXXX)
trap "rm $multiplicities" EXIT

if ! awk '{if(NF!=3){ exit(1); } print $1, $2; } END{if(NR==0) exit(1);}' <$weights_file > $multiplicities; then
  echo "$0: weights file $weights_file has the wrong format."
fi

tot_orig=0
tot_with_multiplicities=0


for f in $data/*.counts; do
  if [ "$f" != "$data/dev.counts" ]; then
    this_tot=$(cat $f | awk '{tot += $2} END{print tot}')
    if ! [ $this_tot -gt 0 ]; then
      echo "$0: there were no counts in counts file $f" 1>&2
      exit 1
    fi
    # weight by the data multiplicity which is the second field of the weights file.
    multiplicity=$(basename $f | sed 's:.counts$::' | utils/apply_map.pl $multiplicities)
    if ! [ $multiplicity -eq $multiplicity ]; then
      echo "$0: error getting multiplicity for data-source $f, check weights file $weights_file"
      exit 1
    fi
    tot_orig=$[tot_orig+this_tot]
    tot_with_multiplicities=$[tot_with_multiplicities+(this_tot*multiplicity)]
  fi

done

if ! [ $tot_orig -gt 0 ]; then
  echo "$0: there was a problem getting counts from directory $data (no counts present?)" 1>&2
  exit 1
fi
if ! [ $tot_with_multiplicities -gt 0 ]; then
  echo "$0: there was a problem getting counts from directory $data (check data-weights file $weights_file)" 1>&2
  exit 1
fi


# adding words_per_split-1 below causes us to round up the number of splits.
num_splits=$[(tot_with_multiplicities+words_per_split-1)/words_per_split]

actual_words_per_split=$[tot_with_multiplicities/num_splits]

if ! [ $num_splits -gt 0 ]; then
  echo "$0: there was a problem getting the number of splits" 1>&2
  exit 1
fi


echo -n "get_num_splits.sh: based on tot-words=$tot_orig (with multiplicities: $tot_with_multiplicities)" 1>&2
echo " and target-words-per-split=$words_per_split, got $num_splits splits, actual words-per-split is $actual_words_per_split" 1>&2

echo $num_splits  # this is the only thing that goes to the standard output.
