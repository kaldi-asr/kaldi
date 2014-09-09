#!/bin/bash

# This script works out the approximate number of frames in a training directory
# this is sometimes needed by higher-level scripts

num_samples=1000


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  echo "Prints the number of frames of data in the data-dir, via sampling rather"
  echo "than trying to access all the data."
fi

data=$1

if [ ! -f $data/feats.scp ]; then
  echo "$0: expected $data/feats.scp to exist"
  exit 1;
fi


sample_frames=$(utils/shuffle_list.pl $data/feats.scp | head -n $num_samples | sort | feat-to-len --print-args=false scp:-)

num_files_orig=$(wc -l <$data/feats.scp)
if [ $num_samples -lt $num_files_orig ]; then
  num_files_sampled=$num_samples
else
  num_files_sampled=$num_files_orig
fi

perl -e "\$n = int(($sample_frames * 1.0 * $num_files_orig) / (1.0 * $num_files_sampled)); print \"\$n\n\";";
