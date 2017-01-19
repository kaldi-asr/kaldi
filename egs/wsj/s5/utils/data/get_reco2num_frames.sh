#! /bin/bash

cmd=run.pl
nj=4

frame_shift=0.01
frame_overlap=0.015

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data>" 
  exit 1
fi

data=$1

if [ -s $data/reco2num_frames ]; then
  echo "$0: $data/reco2num_frames already present!"
  exit 0;
fi

utils/data/get_reco2dur.sh --cmd "$cmd" --nj $nj $data
awk -v fs=$frame_shift -v fovlp=$frame_overlap \
  '{print $1" "int( ($2 - fovlp) / fs)}' $data/reco2dur > $data/reco2num_frames

echo "$0: Computed and wrote $data/reco2num_frames"

