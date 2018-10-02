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
    echo "This works out how many pieces to split a data directory into, and"
    echo "(if just one piece) how many times that piece should be repeated to"
    echo "get the target words-per-split.  A number is printed to the standard"
    echo "output.  If no repeats are necessary it will be the number of splits,"
    echo "a positive number.  If repeats are necessary, then a negative number,"
    echo "interpretable as the negative of the number of times we should repeat"
    echo "the data, is echoed, and the number of splits should be taken to be 1."
    echo "To compute the number of words of training data"
    echo "this script uses <data-dir>/*.counts; they are scaled by the data-multiplicities"
    echo "given as the second field of <weights-file> for each data source."
  ) 1>&2
  exit 1
fi


words_per_split=$1
text=$2
weights_file=$3

! [ $words_per_split -eq $words_per_split ] && \
  echo "$0: first arg must be an integer" 1>&2 && exit 1;

[ ! -d $text ] && \
  echo "$0: no such directory $text" 1>&2 && exit 1;

[ ! -f $weight ] && \
  echo "$0: expected weights file in $weight" 1>&2 && exit 1;

rnnlm/ensure_counts_present.sh $text 1>&2


set -e -o pipefail -u

export LC_ALL=C





multiplicities=$(mktemp tmp.XXXX)
trap "rm $multiplicities" EXIT

if ! awk '{if(NF!=3){ exit(1); } print $1, $2; } END{if(NR==0) exit(1);}' <$weights_file > $multiplicities; then
  echo "$0: weights file $weights_file has the wrong format."
fi

tot_orig=0
tot_with_multiplicities=0


for f in $text/*.counts; do
  if [ "$f" != "$text/dev.counts" ]; then
    this_tot=$(cat $f | awk '{tot += $2} END{printf("%d", tot)}')
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
  echo "$0: there was a problem getting counts from directory $text (no counts present?)" 1>&2
  exit 1
fi
if ! [ $tot_with_multiplicities -gt 0 ]; then
  echo "$0: there was a problem getting counts from directory $text (check data-weights file $weights_file)" 1>&2
  exit 1
fi


# adding words_per_split-1 below causes us to round up the number of splits.
num_splits=$[(tot_with_multiplicities+words_per_split-1)/words_per_split]

actual_words_per_split=$[tot_with_multiplicities/num_splits]

if ! [ $num_splits -gt 0 ]; then
  echo "$0: there was a problem getting the number of splits" 1>&2
  exit 1
fi


num_repeats=$[words_per_split/actual_words_per_split]
if ! [ $num_repeats -ge 1 ]; then
  echo "$0: error computing the number of repeats, got $num_repeats." 1>&2
  exit 1
fi

if [ $num_repeats -gt 1 -a $num_splits -gt 1 ]; then
  echo "$0: script error: both num-repeats and num-splits are over 1." 1>&2
  exit 1
fi

echo -n "get_num_splits.sh: based on tot-words=$tot_orig (with multiplicities: $tot_with_multiplicities)" 1>&2
echo " and target-words-per-split=$words_per_split, got $num_splits splits, actual words-per-split is $actual_words_per_split" 1>&2
if [ $num_repeats -gt 1 ]; then
  echo " ... and num-repeats is $num_repeats" 1>&2
fi


if [ $num_repeats -eq 1 ]; then
  echo $num_splits
else
  echo -$num_repeats
fi
