#!/bin/bash

# This script works out the approximate number of frames in a training directory.
# This is sometimes needed by higher-level scripts


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  (
    echo "Usage: $0 <data-dir>"
    echo "Prints the number of frames of data in the data-dir"
  ) 1>&2
fi

data=$1

if [ ! -f $data/utt2dur ]; then
  utils/data/get_utt2dur.sh $data 1>&2 || exit 1
fi

frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1

awk -v s=$frame_shift '{n += $2} END{printf("%.0f\n", (n / s))}' <$data/utt2dur
