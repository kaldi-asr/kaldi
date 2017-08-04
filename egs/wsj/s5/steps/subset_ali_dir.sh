#!/bin/bash

cmd=run.pl

. path.sh

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  This script creates an alignment directory containing a subset of 
  utterances from the original alignment directory.

  Usage: <subset-data-dir> <data-dir> <ali-dir> <subset-ali-dir>
   e.g.: data/train data/train_sp exp/tri3_ali_sp exp/tri3_ali
EOF
fi

subset_data=$1
data=$2
ali_dir=$3
dir=$4

nj=$(cat $ali_dir/num_jobs) || exit 1
utils/split_data.sh $data $nj

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

rm $dir/ali_tmp.*.{ark,scp} $dir/ali_tmp.scp

exit 0
