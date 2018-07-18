#!/bin/bash

# Copyright 2017  Matthew Maciejewski
# Apache 2.0
#
# Takes a data directory without segmentation but with a vad.scp
# file and outputs a new data directory with segmentation

# begin configuration section.
nj=2
stage=0
cmd=run.pl
segmentation_opts=  # E.g. set this as --segmentation-opts "--silance-proportion 0.2 --max-segment-length 10"
min_duration=0.25

# end configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <in-data-dir> <out-data-dir>"
  echo " Options:"
  echo "    --nj <nj>  # number of parallel jobs"
  echo "    --stage (0|1)  # start script from part-way through"
  echo "    --cmd (run.pl|queue.pl...)  # specify how to run the sub-processes"
  echo "    --segmentation-opts '--opt1 opt1val --opt2 opt2val' # options for segmentation.pl"
  echo "    --min-duration <m> # min duration in seconds for segments (smaller ones are discarded)"
  echo "e.g.:"
  echo "$0 data/train data/train_segmented"
  exit 1;
fi

data=$1
data_out=$2

mkdir -p $data_out || exit 1;
rm $data_out/* 2>/dev/null
mkdir -p $data_out/log

for f in $data/feats.scp $data/vad.scp; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

utils/split_data.sh $data $nj || exit 1;
sdata=$data/split$nj;

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $data_out/log/vad_to_segments.JOB.log \
    copy-vector scp:$sdata/JOB/vad.scp ark,t:- \| \
    sed -e "s/\[ //g;s/ \]//g" \| \
    utils/segmentation.pl $segmentation_opts \
    --remove-noise-only-segments false \
    '>' $sdata/JOB/subsegments || exit 1;

  for n in `seq $nj`; do
    cat $sdata/$n/subsegments
  done | sort | \
  awk -v m=$min_duration '{if ($4 - $3 >= m) { print $0 }}' \
  > $data/subsegments || exit 1;
fi

if [ $stage -le 1 ]; then
  utils/data/subsegment_data_dir.sh $data \
      $data/subsegments $data_out
fi
