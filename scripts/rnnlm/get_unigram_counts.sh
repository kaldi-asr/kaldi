#!/bin/bash

# The script creates text files containing the unigram counts of words.
# things like dev.counts, foo.counts, bar.counts,
# with lines like:
# hello  12341

if [ $# != 1 ]; then
  echo "Usage: rnnlm/get_unigram_counts.sh <data-dir>"
  echo "e.g.: rnnlm/get_unigram_counts.sh data/rnnlm/data"
  echo "This script gets unigram counts of words from data sources, "
  echo "and writes them out in xx.counts files"
  exit 1
fi

data=$1
[ ! -d $data ] && echo "$0: no such directory $data" && exit 1;

set -e -o pipefail -u

export LC_ALL=C

for f in `ls $data/*.txt`; do
  cat $f | sed 's!</S>!</s>!g' | \
      awk '{for(i = 1; i <= NF; i++) {print $i;} print "</s>"}' | sort | uniq -c | \
      awk '{print $2,$1}' > ${f%.*}.counts
done

echo "get_unigram_counts.sh: get counts in $data/*.counts"
