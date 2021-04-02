#!/usr/bin/env bash
#
# Copyright Johns Hopkins University (Author: Daniel Povey) 2016.  Apache 2.0.

# This script performs some analysis of alignments on disk, currently in terms
# of phone lengths, including lengths of leading and trailing silences


# begin configuration section.
cmd=run.pl
#end configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <lang-dir> <ali-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "e.g.:"
  echo "$0 data/lang exp/tri4b"
  echo "This script writes some diagnostics to <ali-dir>/log/alignments.log"
  exit 1;
fi

lang=$1
dir=$2

model=$dir/final.mdl

for f in $lang/words.txt $model $dir/ali.1.gz $dir/num_jobs; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

num_jobs=$(cat $dir/num_jobs) || exit 1

mkdir -p $dir/log

rm $dir/phone_stats.*.gz 2>/dev/null || true

$cmd JOB=1:$num_jobs $dir/log/get_phone_alignments.JOB.log \
  set -o pipefail '&&' ali-to-phones --write-lengths=true "$model"  \
      "ark:gunzip -c $dir/ali.JOB.gz|" ark,t:- \| \
   sed -E 's/^[^ ]+ //' \| \
   awk 'BEGIN{FS=" ; "; OFS="\n";} {print "begin " $1; if (NF>1) print "end " $NF; for (n=1;n<=NF;n++) print "all " $n; }' \| \
   sort \| uniq -c \| gzip -c '>' $dir/phone_stats.JOB.gz || exit 1

if ! $cmd $dir/log/analyze_alignments.log \
  gunzip -c "$dir/phone_stats.*.gz" \| \
  steps/diagnostic/analyze_phone_length_stats.py $lang; then
  echo "$0: analyze_phone_length_stats.py failed, but ignoring the error (it's just for diagnostics)"
fi

grep WARNING $dir/log/analyze_alignments.log
echo "$0: see stats in $dir/log/analyze_alignments.log"

rm $dir/phone_stats.*.gz

exit 0
