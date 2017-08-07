#!/bin/bash

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.

# This script splits an RNNLM data directory in preparation for RNNLM training.
# Note: the split-up data does not have the same format as the original data.
# Suppose we have an RNNLM data-directory
#  data/combined/{foo,bar,dev}.txt
# and we call
#  rnnlm/split_data.sh data/combined 4
# then this script will create:
#  data/combined/split4/{1,2,3,4}.txt
# where these .txt files contain lines like the following:
# head data/combined/split4/1.txt
#  foo  hello my name is al
#  foo  so what are we talking about
#  ...
#  bar  and on the first day, god created the earth
#  ...
#
# so the split-up .txt files contain a mixture of data sources (excluding dev),
# and they contain the name of the data source as the first field of each line.
# Later, during training, we'll randomize the order of the lines in these files
# and use utils/apply_map.pl to turn these data-source names into weights, so
# they will become someting
# like:
#  1.0  hello my name is al
#  0.5  and on the first day, god created the earth
#  1.0  so what are we talking about

# [and these words will then be turned into integer ids before rnnlm-get-egs
# sees them.].


if [ $# != 2 ]; then
  echo "Usage: rnnlm/split_data.sh <data-dir> <num-splits>"
  echo "e.g.: rnnlm/split_data.sh data/text 5"
  echo "This combines and splits the non-dev data sources in <data-dir>,"
  echo "e.g. in the example above, if files data/text/{foo,bar,dev}.txt"
  echo "exist, then this script will distribute foo.txt and bar.txt into"
  echo "file data/text/split5/{1,2,3,4,5}.txt containing lines like"
  echo " foo hello there"
  echo " bar the nasdaq composite index fell for a third consecutive day"
  exit 1
fi

data=$1
num_splits=$2

! [ $num_splits -eq $num_splits ] && \
  echo "$0: <num-splits> must be an integer" && exit 1;

[ ! -d $data ] && \
  echo "$0: no such directory $data" && exit 1;

set -e -o pipefail -u

export LC_ALL=C


data_names=

{
  pushd $data

  #  The following sets data_names to (in the example above) "foo bar".
  for f in *.txt; do
    if [ "$f" != "dev.txt" ]; then
      name=$(echo $f | sed 's:.txt$::')
      [ $name == "" ] && echo "$0: file .txt should not exist" && exit 1
      data_names="$data_names $name"
    fi
  done
  popd
}

if [ -z "$data_names" ]; then
  echo "$0: got no names of data files."
  exit 1;
fi

mkdir -p $data/split$num_splits
# set split_files to e.g. "foo/split10/1.txt foo/split10/2.txt .. foo/split10/10.txt"
split_files=$(for n in $(seq $num_splits); do echo $data/split$num_splits/$n.txt; done)

# make sure all the output files are empty, since distribute_lines.pl appends.
for f in $split_files; do
  echo -n > $f
done

echo "$0: distributing the data"

for name in $data_names; do
  cat $data/$name.txt | rnnlm/internal/distribute_lines.pl --prefix "$name " $split_files
done

echo "$0: done."

exit 0


