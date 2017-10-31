#!/bin/bash

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

transcriptions=$1
wordlist=$2

(
  find $transcriptions -name "*.txt" | xargs egrep -vx '\[[0-9.]+\]'  |cut -f 2- -d ':' | perl -ape 's/ /\n/g;'
) | sort -u | grep -v -E '.*\*.*|<.*>|\(\(\)\)|^-.*|.*-$' > $wordlist

