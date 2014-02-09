#!/bin/bash

# Copyright  2013  Daniel Povey
# Apache 2.0.


if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 trials-file [scores-file]"
  echo "e.g.: $0 data/sre08_trials/short2-short3-female.trials foo"
  exit 1;
fi

trials=$1

[ ! -f $trials ] && echo "Expecting trials file $trials to exist"
if [ $# -eq 2 ]; then
  scores=$2
  tempfile=
else
  tempfile=$(mktemp)
  scores=$tempfile
  cat > $tempfile # put the standard input into tempfile.
fi

echo "Scoring against $trials"

printf '% 12s' 'Condition:'
for condition in $(seq 0 8); do
  printf '% 7d' $condition;
done
echo
printf '% 12s' 'EER:'
eer=$(awk '{print $3}' $scores | paste - $trials | awk '{print $1, $4}' | compute-eer - 2>/dev/null)
printf '% 7.2f' $eer
for condition in $(seq 8); do
  eer=$(awk '{print $3}' $scores | paste - $trials | awk -v c=$condition '{n=4+c; if ($n == "Y") print $1, $4}' | compute-eer - 2>/dev/null)
  printf '% 7.2f' $eer
done
echo


rm $tempfile 2>/dev/null
exit 0;
