#!/bin/bash
# Copyright   2017 Yiwen Shao

nj=4
cmd=run.pl
compress=true
scale_size=40
vertical_shift=10
horizontal_shear=45
augment=false
echo "$0 $@"

. utils/parse_options.sh || exit 1;

data=$1
featdir=$data/data
logdir=$data/log

# make $featdir an absolute pathname
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

if [ -f $data/feats.scp ]; then
    mkdir -p $data/.backup
    echo "$0: moving $data/feats.scp to $data/.backup"
    mv $data/feats.scp $data/.backup
fi

if [ $augment = true ] && [[ $data = *'train'* ]]; then
  if [ ! -d $data/backup ]; then
    mkdir -p $data/backup
    mv $data/text $data/utt2spk $data/images.scp $data/backup/
  else
    cp $data/backup/* $data
  fi
fi


scp=$data/images.scp  
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/images.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;


# add ,p to the input rspecifier so that we can just skip over
# utterances that have bad wave data.
$cmd JOB=1:$nj $logdir/extract_feature.JOB.log \
  local/make_feature_vect.py $logdir --job JOB --scale-size $scale_size --augment $augment --horizontal-shear $horizontal_shear \| \
    copy-feats --compress=$compress --compression-method=7 ark:- \
    ark,scp:$featdir/images.JOB.ark,$featdir/images.JOB.scp \
    || exit 1;  

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featdir/images.$n.scp || exit 1;
done > $data/feats.scp || exit 1

# re-map utt2spk, images.scp and text if doing image augmentation
# on training set
if [ $augment = true ] && [[ $data = *'train'* ]]; then
  local/process_augment_data.py $data
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt 
fi    

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
    echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi
