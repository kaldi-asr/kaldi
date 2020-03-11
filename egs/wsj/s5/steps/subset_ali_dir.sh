#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

cmd=run.pl

if [ -f ./path.sh ]; then . ./path.sh; fi

. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  This script creates an alignment directory containing a subset of 
  utterances contained in <subset-data-dir> from the 
  original alignment directory containing alignments for utterances in
  <full-data-dir>.

  The number of split jobs in the output alignment directory is 
  equal to the number of jobs in the original alignment directory, 
  unless the subset data directory has too few speakers.

  Usage: $0 [options] <full-data-dir> <subset-data-dir> <ali-dir> <subset-ali-dir>
   e.g.: $0 data/train_sp data/train exp/tri3_ali_sp exp/tri3_ali

  Options: 
      --cmd (utils/run.pl|utils/queue.pl <queue opts>)  # how to run jobs.
EOF
  exit 1
fi

data=$1
subset_data=$2
ali_dir=$3
dir=$4

nj=$(cat $ali_dir/num_jobs) || exit 1
utils/split_data.sh $data $nj

mkdir -p $dir
cp $ali_dir/{final.mdl,*.mat,*_opts,tree} $dir/ || true
cp -r $ali_dir/phones $dir 2>/dev/null || true

$cmd JOB=1:$nj $dir/log/copy_alignments.JOB.log \
  copy-int-vector "ark:gunzip -c $ali_dir/ali.JOB.gz |" \
  ark,scp:$dir/ali_tmp.JOB.ark,$dir/ali_tmp.JOB.scp || exit 1

for n in `seq $nj`; do
  cat $dir/ali_tmp.$n.scp 
done > $dir/ali_tmp.scp

num_spk=$(cat $subset_data/spk2utt | wc -l)
if [ $num_spk -lt $nj ]; then
  nj=$num_spk
fi

utils/split_data.sh $subset_data $nj
$cmd JOB=1:$nj $dir/log/filter_alignments.JOB.log \
  copy-int-vector \
  "scp:utils/filter_scp.pl $subset_data/split${nj}/JOB/utt2spk $dir/ali_tmp.scp |" \
  "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1

echo $nj > $dir/num_jobs

rm $dir/ali_tmp.*.{ark,scp} $dir/ali_tmp.scp

exit 0
