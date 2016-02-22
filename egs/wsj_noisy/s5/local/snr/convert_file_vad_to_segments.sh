#!/bin/bash

. path.sh

cmd=run.pl
nj=4
keep_only_speech=false

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <vad-dir> <dir>"
  echo " e.g.: $0 exp/vad_data_prep_dev/file_vad exp/vad_data_prep_dev/file_vad"
  exit 1
fi

vad_dir=$1
dir=$2

merge_opts=( --remove-labels=4:10 --merge-labels=0:1 --merge-dst-label=1 )
if $keep_only_speech; then
  merge_opts=( --remove-labels=0:4:10 )
fi

$cmd JOB=1:$nj $dir/log/get_segments.JOB.log \
  segmentation-init-from-ali ark:$vad_dir/vad.JOB.ark ark:- \| \
  segmentation-post-process ${merge_opts[@]} \
  --shrink-length=20 --shrink-label=1 --merge-adjacent-segments=true --max-intersegment-length=1 \
  ark:- ark:- \| segmentation-to-segments --single-speaker=true ark:- \
  ark,t:$dir/reco2utt.JOB ark,t:$dir/segments.JOB || exit 1

for n in `seq $nj`; do
  cat $dir/reco2utt.$n
done > $dir/reco2utt

for n in `seq $nj`; do
  cat $dir/segments.$n
done > $dir/segments

