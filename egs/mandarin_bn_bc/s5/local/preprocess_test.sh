#!/usr/bin/env bash

. ./path.sh

set -e
nj=8

max_segment_duration=30
overlap_duration=5
max_remaining_duration=15

. utils/parse_options.sh

data=$1
seg_data=$2

if [ ! -s $data/utt2dur ]; then
  utils/data/get_utt2dur.sh --nj $nj ${data} 1>&2 || exit 1;
fi

if [ ! -f $data/segments ]; then
  utils/data/get_segments_for_data.sh ${data} > ${data}/segments || exit 1;
fi

utils/data/get_uniform_subsegments.py --max-segment-duration=${max_segment_duration} \
  --overlap-duration=${overlap_duration} --max-remaining-duration=${max_remaining_duration} \
  ${data}/segments > ${data}/uniform_sub_segments || exit 1;

utils/data/subsegment_data_dir.sh ${data} ${data}/uniform_sub_segments $seg_data || exit 1;

echo "$0: ${data} segmented"

exit 0;
