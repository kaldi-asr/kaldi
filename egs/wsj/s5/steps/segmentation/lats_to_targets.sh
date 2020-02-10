#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script converts lattices into targets for training neural network
# for speech activity detection. The targets is a matrix of size 
# (num-frames-subsampled x 3)
# with each row representing probabilities for speech, silence and 
# garbage classes for the corresponding frame (after subsampling). The 
# probability values are lattice posteriors for the 3 classes and are
# obtained by summing up phone arc posteriors for the phones
# corresponding to each class.
# The mapping from phones to speech / silence / garbage classes
# is defined by the options --silence-phones and --garbage-phones.
# Also "speech" phones longer than --max-phone-duration seconds are 
# treated as "garbage".

set -o pipefail

silence_phones=
garbage_phones=
max_phone_duration=0.5
acwt=0.1

cmd=run.pl

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  This script converts lattices into targets for training neural network
  for speech activity detection. The targets is a matrix of size 
  (num-frames-subsampled x 3)
  with each row representing probabilities for speech, silence and 
  garbage classes for the corresponding frame (after subsampling). The 
  probability values are lattice posteriors for the 3 classes and are
  obtained by summing up phone arc posteriors for the phones
  corresponding to each class.
  The mapping from phones to speech / silence / garbage classes
  is defined by the options --silence-phones and --garbage-phones.
  Also "speech" phones longer than --max-phone-duration seconds are 
  treated as "garbage".

  Usage: steps/segmentation/lats_to_targets.sh <data-dir> <lang> <lattice-dir> <targets-dir>"
  e.g.: steps/segmentation/lats_to_targets.sh \
  --silence-phones exp/segmentation1a/silence_phones.txt \
  --garbage-phones exp/segmentation1a/garbage_phones.txt \
  --max-phone-duration 0.5 \
  data/train_split10s data/lang \
  exp/segmentation1a/tri3b_train_split10s_lats \
  exp/segmentation1a/tri3b_train_split10s_targets

  note: 
  silence_phones.txt and garbage_phones.txt must list phones, one per line.
  garbage_phones.txt can contain phones corresponding to ambiguous items like 
  OOV, laugh and spoken noise that you want to map to "garbage class".
  silence_phones.txt might just contain the phones from 
  data/lang/phones/silence_phones.txt other than the garbage phones. These
  are mapped to the "silence" class.
EOF
  exit 1
fi

data=$1
lang=$2
lats_dir=$3
dir=$4

if [ -f $lats_dir/final.mdl ]; then
  srcdir=$lats_dir
else
  srcdir=$lats_dir/..
fi

for f in $data/utt2spk $lats_dir/lat.1.gz $srcdir/final.mdl; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

mkdir -p $dir

if [ -z "$garbage_phones" ]; then
  oov_phone=$(steps/segmentation/internal/get_oov_phone.py $lang) || exit 1
  echo $oov_phone | utils/int2sym.pl $lang/phones.txt > $dir/garbage_phones.txt || exit 1
else 
  cp $garbage_phones $dir/garbage_phones.txt || exit 1
fi

if [ -z "$silence_phones" ]; then
  cat $lang/silence_phones.txt | \
    utils/filter_scp.pl --exclude $dir/garbage_phones.txt > \
    $dir/silence_phones.txt
else 
  cp $silence_phones $dir/silence_phones.txt
fi

nj=$(cat $lats_dir/num_jobs) || exit 1

$cmd JOB=1:$nj $dir/log/get_arc_info.JOB.log \
  lattice-push "ark:gunzip -c $lats_dir/lat.JOB.gz |" ark:- \| \
  lattice-align-phones --replace-output-symbols=true $srcdir/final.mdl ark:- ark:- \| \
  lattice-arc-post --acoustic-scale=$acwt $srcdir/final.mdl ark:- - \| \
  utils/int2sym.pl -f 5 $lang/phones.txt '>' \
  $dir/arc_info_sym.JOB.txt || exit 1

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

frame_subsampling_factor=1
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $srcdir/frames_subsampling_factor)
  echo $frame_subsampling_factor > $dir/frame_subsampling_factor
fi

frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1
max_phone_len=$(perl -e "print int($max_phone_duration / $frame_shift)")

$cmd JOB=1:$nj $dir/log/get_targets.JOB.log \
  steps/segmentation/internal/arc_info_to_targets.py \
    --silence-phones=$dir/silence_phones.txt \
    --garbage-phones=$dir/garbage_phones.txt \
    --max-phone-length=$max_phone_len \
    $dir/arc_info_sym.JOB.txt - \| \
  copy-feats ark,t:- \
    ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

for n in $(seq $nj); do
  cat $dir/targets.$n.scp
done > $dir/targets.scp

steps/segmentation/validate_targets_dir.sh $dir $data || exit 1

echo "$0: Done creating targets in $dir/targets.scp"
