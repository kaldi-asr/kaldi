#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

srcdir=data/local/data
tmpdir=data/local/

for t in train; do 
  utils/fix_data_dir.sh $srcdir/$t
  utils/copy_data_dir.sh $srcdir/$t data/$t
  cat $srcdir/$t/text | \
    local/normalize_transcripts.pl "<NOISE>" "<SPOKEN_NOISE>" > \
    data/$t/text
  utils/fix_data_dir.sh data/$t
done

for t in eval98 eval98.pem; do
  utils/copy_data_dir.sh $srcdir/$t data/$t
  utils/fix_data_dir.sh data/$t
done


