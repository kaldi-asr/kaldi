#!/usr/bin/env bash

cmd=queue.pl
nj=40
stage=0
speed_perturb=true

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <utt-map> <data-dir> <src-lat-dir> <out-lat-dir>"
  exit 1
fi

utt_map=$1
data=$2
srcdir=$3
dir=$4

mkdir -p $dir

cp $srcdir/{phones.txt,tree,final.mdl} $dir || exit 1
cp $srcdir/{final.alimdl,final.occs,splice_opts,cmvn_opts,delta_opts,final.mat,full.mat} 2>/dev/null || true

nj_src=$(cat $srcdir/num_jobs) || exit 1

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj_src $dir/log/copy_lats_orig.JOB.log \
    lattice-copy "ark:gunzip -c $srcdir/lat.JOB.gz |" \
    ark,scp:$dir/lat_orig.JOB.ark,$dir/lat_orig.JOB.scp || exit 1
fi

for n in $(seq $nj_src); do
  cat $dir/lat_orig.$n.scp
done > $dir/lat_orig.scp || exit 1

if $speed_perturb; then
  for s in 0.9 1.1; do
    awk -v s=$s '{print "sp"s"-"$1" sp"s"-"$2}' $utt_map
  done | cat - $utt_map | sort -k1,1 > $dir/utt_map
  utt_map=$dir/utt_map
fi

if [ $stage -le 2 ]; then
  utils/filter_scp.pl -f 2 $dir/lat_orig.scp < $utt_map | \
    utils/apply_map.pl -f 2 $dir/lat_orig.scp > \
    $dir/lat.scp || exit 1

  if [ ! -s $dir/lat.scp ]; then
    echo "$0: $dir/lat.scp is empty. Something went wrong!"
    exit 1
  fi
fi

utils/split_data.sh $data $nj

if [ $stage -le 3 ]; then
  $cmd JOB=1:$nj $dir/log/copy_lats.JOB.log \
    lattice-copy "scp:utils/filter_scp.pl $data/split$nj/JOB/utt2spk $dir/lat.scp |" \
    "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1
fi

echo $nj > $dir/num_jobs

if [ -f $srcdir/ali.1.gz ]; then
  if [ $stage -le 4 ]; then
    $cmd JOB=1:$nj_src $dir/log/copy_ali_orig.JOB.log \
      copy-int-vector "ark:gunzip -c $srcdir/ali.JOB.gz |" \
      ark,scp:$dir/ali_orig.JOB.ark,$dir/ali_orig.JOB.scp || exit 1
  fi

  for n in $(seq $nj_src); do
    cat $dir/ali_orig.$n.scp
  done > $dir/ali_orig.scp || exit 1

  if [ $stage -le 5 ]; then
    utils/filter_scp.pl -f 2 $dir/ali_orig.scp < $utt_map | \
      utils/apply_map.pl -f 2 $dir/ali_orig.scp > \
      $dir/ali.scp || exit 1
  
    if [ ! -s $dir/ali.scp ]; then
      echo "$0: $dir/ali.scp is empty. Something went wrong!"
      exit 1
    fi
  fi

  utils/split_data.sh $data $nj

  if [ $stage -le 6 ]; then
    $cmd JOB=1:$nj $dir/log/copy_ali.JOB.log \
      copy-int-vector "scp:utils/filter_scp.pl $data/split$nj/JOB/utt2spk $dir/ali.scp |" \
      "ark:|gzip -c > $dir/ali.JOB.gz" || exit 1
  fi
fi

rm $dir/lat_orig.*.{ark,scp} $dir/ali_orig.*.{ark,scp} 2>/dev/null || true
