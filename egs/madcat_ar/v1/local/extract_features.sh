#!/bin/bash
# Copyright   2017 Yiwen Shao
#             2018 Ashish Arora

nj=4
cmd=run.pl
feat_dim=40
echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

data=$1
featdir=$data/data
scp=$data/images.scp
logdir=$data/log

mkdir -p $logdir
mkdir -p $featdir

# make $featdir an absolute pathname
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

for n in $(seq $nj); do
    split_scps="$split_scps $logdir/images.$n.scp"
done

# split images.scp
utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $logdir/extract_features.JOB.log \
  local/make_features.py $logdir/images.JOB.scp \
    --allowed_len_file_path $data/allowed_lengths.txt \
    --feat-dim $feat_dim \| \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:$featdir/images.JOB.ark,$featdir/images.JOB.scp

## aggregates the output scp's to get feats.scp
for n in $(seq $nj); do
  cat $featdir/images.$n.scp || exit 1;
done > $data/feats.scp || exit 1
