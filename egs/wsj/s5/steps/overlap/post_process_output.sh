#!/usr/bin/env bash

# Copyright 2015-17  Vimal Manohar
#           2020     Desh Raj
# Apache 2.0.

# This script post-processes the output of the overlap neural network,
# which is in the form of frame-level alignments, into an RTTM file.
# The alignments must be 0/1/2 denoting silence/single/overlap. Based
# on this, this script can also be used to get single speaker regions.

set -e -o pipefail -u
. ./path.sh

cmd=run.pl
stage=-10
nj=18

region_type=overlap # change this to "single" to get only single-speaker regions

# The values below are in seconds
frame_shift=0.01
segment_padding=0.2
min_segment_dur=0
merge_consecutive_max_dur=inf

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "This script post-processes the output of steps/segmentation/decode_sad.sh, "
  echo "which is in the form of frame-level alignments, into kaldi segments. "
  echo "The alignments must be speech activity detection marks i.e. 1 for silence "
  echo "and 2 for speech."
  echo "Usage: $0 <data-dir> <output-dir> <rttm-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire"
  exit 1
fi

data_dir=$1
output_dir=$2    # Alignment directory containing frame-level SAD labels
dir=$3

mkdir -p $dir

for f in $output_dir/ali.1.gz $output_dir/num_jobs; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f" && exit 1
  fi
done

nj=`cat $output_dir/num_jobs` || exit 1
utils/split_data.sh $data_dir $nj

utils/data/get_utt2dur.sh $data_dir

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
    copy-int-vector "ark:gunzip -c $output_dir/ali.JOB.gz |" ark,t:- \| \
    steps/overlap/output_to_rttm.py \
      --region-type=$region_type \
      --frame-shift=$frame_shift --segment-padding=$segment_padding \
      --min-segment-dur=$min_segment_dur --merge-consecutive-max-dur=$merge_consecutive_max_dur \
      --utt2dur=$data_dir/utt2dur - $dir/rttm_${region_type}.JOB
fi

echo $nj > $dir/num_jobs

for n in $(seq $nj); do 
  cat $dir/rttm_${region_type}.$n
done > $dir/rttm_${region_type}
