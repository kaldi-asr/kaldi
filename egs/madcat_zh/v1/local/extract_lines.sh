#!/bin/bash
# Copyright   2018 Ashish Arora

nj=4
cmd=run.pl
download_dir=/export/corpora/LDC/LDC2014T13
dataset_file=data/download/datasplits/madcat.dev.raw.lineid
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

data=$1
log_dir=$data/log
mkdir -p $log_dir
mkdir -p $data

for n in $(seq $nj); do
    split_scps="$split_scps $log_dir/lines.$n.scp"
done

utils/split_scp.pl $dataset_file $split_scps || exit 1;

for n in $(seq $nj); do
  mkdir -p $data/$n
done

$cmd JOB=1:$nj $log_dir/extract_lines.JOB.log \
  local/create_line_image_from_page_image.py $download_dir $log_dir/lines.JOB.scp $data/JOB \
  || exit 1;

## concatenate the .scp files together.
for n in $(seq $nj); do
  cat $data/$n/images.scp || exit 1;
done > $data/images.scp || exit 1
