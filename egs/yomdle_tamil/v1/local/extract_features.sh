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

if [ -f $data/feats.scp ]; then
    echo "$0: ERROR: $data/feats.scp should not exist"
fi

if [ ! -d $data/backup ]; then
  mkdir -p $data/backup
  mv $data/text $data/utt2spk $data/images.scp $data/backup/
else
  cp $data/backup/* $data
fi

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

# re-map utt2spk, images.scp and text if doing image augmentation
local/process_augment_data.py $data
utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt 

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
    echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi
