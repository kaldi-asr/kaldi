#!/bin/bash

noise_list="reverb1:babble:music:noise"
max_jobs_run=50
nj=100
cmd=queue.pl
write_binary=true

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <out-data> <src-ali-dir> <out-ali-dir>"
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
utt_prefixes=`echo $noise_list | awk -F ":" '{for (i=1; i<=NF; i++) printf "%s- ", $i}'`
for p in $utt_prefixes; do
  cat $dir/ali_tmp.*.scp | awk -v p=$p '{print p$0}'
done | sort -k1,1 > $dir/ali_out.scp.noise

cat $dir/ali_tmp.*.scp | awk '{print $0}' | sort -k1,1 > $dir/ali_out.scp.clean

cat $dir/ali_out.scp.clean $dir/ali_out.scp.noise | sort -k1,1 > $dir/ali_out.scp

utils/split_data.sh ${data} $nj

# Copy and dump the lattices for perturbed data
echo Creating alignments for augmented data by copying alignments from clean data
$cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/copy_out_ali.JOB.log \
  copy-int-vector --binary=$write_binary \
  "scp:utils/filter_scp.pl ${data}/split$nj/JOB/utt2spk $dir/ali_out.scp |" \
  "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1

#rm $dir/ali_out.scp.{prefix,clean} $dir/ali_out.scp
rm $dir/ali_tmp.*

echo $nj > $dir/num_jobs

for f in cmvn_opts tree splice_opts phones.txt final.mdl splice_opts tree frame_subsampling_factor; do
  if [ -f $src_dir/$f ]; then cp $src_dir/$f $dir/$f; fi
done
