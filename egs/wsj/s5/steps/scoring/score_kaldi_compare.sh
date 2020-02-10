#!/usr/bin/env bash
# Copyright 2016 Nicolas Serrano
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
replications=10000
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <score-dir1> <score-dir2> <score-compare-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --replications <int>            # number of bootstrap evaluation to compute confidence."
  exit 1;
fi

dir1=$1
dir2=$2
dir_compare=$3

mkdir -p $dir_compare/log

for d in $dir1 $dir2; do
  for f in test_filt.txt best_wer; do
    [ ! -f $d/$f ] && echo "$0: no such file $d/$f" && exit 1;
  done
done


best_wer_file1=$(awk '{print $NF}' $dir1/best_wer)
best_transcript_file1=$(echo $best_wer_file1 | sed -e 's=.*/wer_==' | \
        awk -v FS='_' -v dir=$dir1 '{print dir"/penalty_"$2"/"$1".txt"}')

best_wer_file2=$(awk '{print $NF}' $dir2/best_wer)
best_transcript_file2=$(echo $best_wer_file2 | sed -e 's=.*/wer_==' | \
        awk -v FS='_' -v dir=$dir2 '{print dir"/penalty_"$2"/"$1".txt"}')

$cmd $dir_compare/log/score_compare.log \
  compute-wer-bootci --replications=$replications \
    ark:$dir1/test_filt.txt ark:$best_transcript_file1 ark:$best_transcript_file2 \
    '>' $dir_compare/wer_bootci_comparison || exit 1;

exit 0;
