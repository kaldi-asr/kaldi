#!/bin/bash
# Copyright   2018 Ashish Arora

nj=4
cmd=run.pl
download_dir1=/export/corpora/LDC/LDC2012T15/data
download_dir2=/export/corpora/LDC/LDC2013T09/data
download_dir3=/export/corpora/LDC/LDC2013T15/data
train_split_file=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.train.raw.lineid
test_split=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.test.raw.lineid
dev_split=/home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.dev.raw.lineid
lines_dir=data/local/lines
logdir=data/local/log
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

data=$1
featdir=$data/data
scp=$data/images.scp
logdir=$data/log
scp=$dev_split

mkdir -p $logdir
mkdir -p $featdir

# make $featdir an absolute pathname
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`
lines_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

for n in $(seq $nj); do
    split_scps="$split_scps $logdir/lines.$n.scp"
done

# split images.scp
utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $logdir/extract_lines.JOB.log \
  local/create_line_image_from_page_image.py $download_dir1 $download_dir2 $download_dir3 $logdir data/local/lines --job JOB \| \
    scp:$featdir/images.JOB.scp \
    || exit 1;  

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featdir/images.$n.scp || exit 1;
done > $data/feats.scp || exit 1
