#!/bin/bash
# Copyright 2018-2020  Daniel Povey
#           2018-2020  Yiming Wang

utt_prefixes=
max_jobs_run=30
nj=75
cmd=run.pl
write_compact=true

echo "$0 $@"  # Print the command line for logging 

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <out-data> <src-lat-dir> <out-lat-dir>"
  exit 1
fi

data=$1
src_dir=$2
dir=$3

rm -rf $dir 2>/dev/null
cp -r $src_dir $dir

num_jobs=$(cat $src_dir/num_jobs)

rm -f $dir/lat_tmp.*.{ark,scp} 2>/dev/null

# Copy the lattices temporarily
$cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs $dir/log/copy_lattices.JOB.log \
  lattice-copy --write-compact=$write_compact \
  "ark:gunzip -c $src_dir/lat.JOB.gz |" \
  ark,scp:$dir/lat_tmp.JOB.ark,$dir/lat_tmp.JOB.scp || exit 1

# Make copies of utterances for perturbed data
for p in $utt_prefixes; do
  cat $dir/lat_tmp.*.scp | local/add_prefix_to_scp.py --prefix $p
done >$dir/lat_out.scp.tmp
cat $dir/lat_tmp.*.scp $dir/lat_out.scp.tmp | sort -k1,1 >$dir/lat_out.scp
rm -f $dir/lat_out.scp.tmp 2>/dev/null

utils/split_data.sh ${data} $nj

# Copy and dump the lattices for perturbed data
$cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/copy_out_lattices.JOB.log \
  lattice-copy --write-compact=$write_compact \
  "scp:utils/filter_scp.pl ${data}/split$nj/JOB/utt2spk $dir/lat_out.scp |" \
  "ark:| gzip -c > $dir/lat.JOB.gz" || exit 1

rm $dir/lat_tmp.* #$dir/lat_out.scp

echo $nj > $dir/num_jobs
