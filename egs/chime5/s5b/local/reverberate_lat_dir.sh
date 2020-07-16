#!/usr/bin/env bash

# Copyright 2018  Vimal Manohar
# Apache 2.0

num_data_reps=1
cmd=run.pl
nj=20
include_clean=false

. utils/parse_options.sh
. ./path.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <train-data-dir> <noisy-latdir> <clean-latdir> <output-latdir>"
  exit 1
fi

train_data_dir=$1
noisy_latdir=$2
clean_latdir=$3
dir=$4

clean_nj=$(cat $clean_latdir/num_jobs)

$cmd JOB=1:$clean_nj $dir/copy_clean_lattices.JOB.log \
  lattice-copy "ark:gunzip -c $clean_latdir/lat.JOB.gz |" \
  ark,scp:$dir/lats_clean.JOB.ark,$dir/lats_clean.JOB.scp || exit 1
  
for n in $(seq $clean_nj); do
  cat $dir/lats_clean.$n.scp 
done > $dir/lats_clean.scp

for i in $(seq $num_data_reps); do
  cat $dir/lats_clean.scp | awk -vi=$i '{print "rev"i"_"$0}'
done > $dir/lats_rvb.scp

noisy_nj=$(cat $noisy_latdir/num_jobs)
$cmd JOB=1:$noisy_nj $dir/copy_noisy_lattices.JOB>log \
  lattice-copy "ark:gunzip -c $noisy_latdir/lat.JOB.gz |" \
  ark,scp:$dir/lats_noisy.JOB.ark,$dir/lats_noisy.JOB.scp || exit 1

optional_clean=
if $include_clean; then
  optional_clean=$dir/lats_clean.scp
fi

for n in $(seq $noisy_nj); do
  cat $dir/lats_noisy.$n.scp
done | cat - $dir/lats_rvb.scp ${optional_clean} | sort -k1,1 > $dir/lats.scp

utils/split_data.sh $train_data_dir $nj
$cmd JOB=1:$nj $dir/copy_lattices.JOB.log \
  lattice-copy "scp:utils/filter_scp.pl $train_data_dir/split$nj/JOB/utt2spk $dir/lats.scp |" \
  "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1

echo $nj > $dir/num_jobs

if [ -f $clean_latdir/ali.1.gz ]; then
  $cmd JOB=1:$clean_nj $dir/copy_clean_alignments.JOB.log \
    copy-int-vector "ark:gunzip -c $clean_latdir/ali.JOB.gz |" \
    ark,scp:$dir/ali_clean.JOB.ark,$dir/ali_clean.JOB.scp
    
  for n in $(seq $clean_nj); do
    cat $dir/ali_clean.$n.scp 
  done > $dir/ali_clean.scp

  for i in $(seq $num_data_reps); do
    cat $dir/ali_clean.scp | awk -vi=$i '{print "rev"i"_"$0}'
  done > $dir/ali_rvb.scp
  
  optional_clean=
  if $include_clean; then
    optional_clean=$dir/ali_clean.scp
  fi

  $cmd JOB=1:$noisy_nj $dir/copy_noisy_alignments.JOB.log \
    copy-int-vector "ark:gunzip -c $noisy_latdir/ali.JOB.gz |" \
    ark,scp:$dir/ali_noisy.JOB.ark,$dir/ali_noisy.JOB.scp

  for n in $(seq $noisy_nj); do
    cat $dir/ali_noisy.$n.scp
  done | cat - $dir/ali_rvb.scp $optional_clean | sort -k1,1 > $dir/ali.scp

  utils/split_data.sh $train_data_dir $nj || exit 1
  $cmd JOB=1:$nj $dir/copy_rvb_alignments.JOB.log \
    copy-int-vector "scp:utils/filter_scp.pl $train_data_dir/split$nj/JOB/utt2spk $dir/ali.scp |" \
    "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1
fi

cp $clean_latdir/{final.*,tree,*.mat,*opts,*.txt} $dir || true

rm $dir/lats_{clean,noisy}.*.{ark,scp} $dir/ali_{clean,noisy}.*.{ark,scp} || true # save space
