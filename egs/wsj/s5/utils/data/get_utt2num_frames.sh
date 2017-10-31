#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

cmd=run.pl
nj=4

frame_shift=0.01
frame_overlap=0.015

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "This script writes a file utt2num_frames with the "
  echo "number of frames in each utterance as measured based on the "
  echo "duration of the utterances (in utt2dur) and the specified "
  echo "frame_shift and frame_overlap."
  echo "Usage: $0 <data>" 
  exit 1
fi

data=$1

if [ -s $data/utt2num_frames ]; then
  echo "$0: $data/utt2num_frames already present!"
  exit 0;
fi

if [ ! -f $data/feats.scp ]; then
  utils/data/get_utt2dur.sh $data
  awk -v fs=$frame_shift -v fovlp=$frame_overlap \
    '{print $1" "int( ($2 - fovlp) / fs)}' $data/utt2dur > $data/utt2num_frames
  exit 0
fi

utils/split_data.sh --per-utt $data $nj || exit 1
$cmd JOB=1:$nj $data/log/get_utt2num_frames.JOB.log \
  feat-to-len scp:$data/split${nj}utt/JOB/feats.scp ark,t:$data/split${nj}utt/JOB/utt2num_frames || exit 1

for n in `seq $nj`; do
  cat $data/split${nj}utt/$n/utt2num_frames
done > $data/utt2num_frames

echo "$0: Computed and wrote $data/utt2num_frames"
