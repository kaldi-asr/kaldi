#! /bin/bash

# Copyright 2015-17  Vimal Manohar
# Apache 2.0.

# This script post-processes the output of steps/segmentation/decode_sad.sh,
# which is in the form of frame-level alignments, into kaldi segments.

set -e -o pipefail -u
. path.sh

cmd=run.pl
stage=-10

frame_shift=0.01
post_processing_opts="--segment-padding=0.2 --max-intersegment-duration=0.3 --min-segment-duration=0.3 --max-segment-duration=10.0 --overlap-duration=1.0 --max-remaining-duration=2.0"
nj=18

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "This script post-processes the output of steps/segmentation/decode_sad.sh, "
  echo "which is in the form of frame-level alignments, into kaldi segments. "
  echo "Usage: $0 <data-dir> <vad-dir> <segmentation-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire"
  exit 1
fi

data_dir=$1
vad_dir=$2    # Alignment directory containing frame-level labels
dir=$3

mkdir -p $dir

for f in $vad_dir/ali.1.gz $vad_dir/num_jobs; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f" && exit 1
  fi
done

nj=`cat $vad_dir/num_jobs` || exit 1
utils/split_data.sh $data_dir $nj

utils/data/get_utt2dur.sh $data_dir

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
    copy-int-vector "ark:gunzip -c $vad_dir/ali.JOB.gz |" ark,t:- \| \
    steps/segmentation/internal/ali_to_segments.py \
      --frame-shift=$frame_shift $post_processing_opts \
      --utt2dur=$data_dir/utt2dur - $dir/segments.JOB
fi

echo $nj > $dir/num_jobs

for n in $(seq $nj); do 
  cat $dir/segments.$n
done > $dir/segments
