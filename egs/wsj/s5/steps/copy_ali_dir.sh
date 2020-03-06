#!/usr/bin/env bash
# Copyright 2019   Phani Sankar Nidadavolu
# Apache 2.0.

prefixes="reverb1 babble music noise"
include_original=true
max_jobs_run=50
nj=100
cmd=queue.pl
write_binary=true

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <out-data> <src-ali-dir> <out-ali-dir>"
  echo "This script creates alignments for the aug dirs by copying "
  echo " the alignments of original train dir"
  echo "While copying it adds prefix to the utterances specified by prefixes option"
  echo "Note that the original train dir does not have any prefix"
  echo "To include the original training directory in the copied "
  echo "version set the --include-original option to true"
  echo "main options (for others, see top of script file)"
  echo "  --prefixes <string of prefixes to add>    # All the prefixes of aug data to be included"
  echo "  --include-original <true/false>           # If true, will copy the alignements of original dir"
  echo "  --write-compact <true/false>              # Write lattices in compact mode"
  exit 1
fi

data=$1
src_dir=$2
dir=$3

mkdir -p $dir

num_jobs=$(cat $src_dir/num_jobs)

rm -f $dir/ali_tmp.*.{ark,scp} 2>/dev/null

# Copy the alignments temporarily
echo "creating temporary alignments in $dir"
$cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs $dir/log/copy_ali_temp.JOB.log \
  copy-int-vector --binary=$write_binary \
  "ark:gunzip -c $src_dir/ali.JOB.gz |" \
  ark,scp:$dir/ali_tmp.JOB.ark,$dir/ali_tmp.JOB.scp || exit 1

# Make copies of utterances for perturbed data
for p in $prefixes; do
  cat $dir/ali_tmp.*.scp | awk -v p=$p '{print p"-"$0}'
done | sort -k1,1 > $dir/ali_out.scp.aug

if [ "$include_original" == "true" ]; then
  cat $dir/ali_tmp.*.scp | awk '{print $0}' | sort -k1,1 > $dir/ali_out.scp.clean
  cat $dir/ali_out.scp.clean $dir/ali_out.scp.aug | sort -k1,1 > $dir/ali_out.scp
else
  cat $dir/ali_out.scp.aug | sort -k1,1 > $dir/ali_out.scp
fi

utils/split_data.sh ${data} $nj

# Copy and dump the lattices for perturbed data
echo Creating alignments for augmented data by copying alignments from clean data
$cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/copy_out_ali.JOB.log \
  copy-int-vector --binary=$write_binary \
  "scp:utils/filter_scp.pl ${data}/split$nj/JOB/utt2spk $dir/ali_out.scp |" \
  "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1

rm $dir/ali_out.scp.{aug,clean} $dir/ali_out.scp
rm $dir/ali_tmp.*

echo $nj > $dir/num_jobs

for f in cmvn_opts tree splice_opts phones.txt final.mdl splice_opts tree frame_subsampling_factor; do
  if [ -f $src_dir/$f ]; then cp $src_dir/$f $dir/$f; fi
done
