#!/usr/bin/env bash

# This script makes sure that a <text-dir>, as validated by validate_text_dir.py,
# has unigram counts present (*.counts).


if [ $# != 1 ]; then
  echo "Usage: $0 <text-dir>"
  echo "Makes sure unigram counts (*.counts) are present in <text-dir>,"
  echo "and if not, sets them up."
  exit 1;
fi


dir=$1

for f in `ls $dir/*.txt`; do
  counts_file=$(echo $f | sed s/.txt$/.counts/)
  if [ ! -f $counts_file -o $counts_file -ot $f ]; then
    echo "$0: generating counts file for $f" 1>&2
    cat $f | sed 's/ \+/ /g' | python -c '
import sys
counts = dict()
eos = "</s>"
counts[eos] = 0
for line in sys.stdin:
    tokens = line.strip("\n").strip(" ").split(" ")
    for token in tokens:
        if token not in counts:
            counts[token] = 0
        counts[token] += 1
    counts[eos] += 1
for word, count in counts.items():
    print(f"{word} {count}")
' | sort > $counts_file
  fi
done
