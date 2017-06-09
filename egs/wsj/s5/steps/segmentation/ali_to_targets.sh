#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

set -o pipefail

silence_words=
garbage_words=
max_phone_duration=0.5
acwt=0.1

cmd=run.pl

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  Usage: steps/segmentation/ali_to_targets.sh <data-dir> <lang> <lattice-dir> <targets-dir>"
  e.g.: steps/segmentation/ali_to_targets.sh \
  --silence-words exp/segmentation1a/silence_words.txt \
  --garbage-words exp/segmentation1a/garbage_words.txt \
  --max-phone-duration 0.5 \
  data/train_split10s data/lang \
  exp/segmentation1a/tri3b_train_split10s_ali \
  exp/segmentation1a/tri3b_train_split10s_targets

  note: silence_words.txt might just contain <eps> (which is how lattice-arc-post will
  print optional-silence) or might contain other stuff too; garbage_words.txt will
  contain a list of words that we consider neither speech nor silence.  This is
  quite setup dependent.  E.g. in aspire it would contain "mm".
EOF
  exit 1
fi

data=$1
lang=$2
ali_dir=$3
dir=$4

if [ -f $ali_dir/final.mdl ]; then
  srcdir=$ali_dir
else
  srcdir=$ali_dir/..
fi

for f in $data/utt2spk $ali_dir/ali.1.gz $srcdir/final.mdl; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

mkdir -p $dir

if [ -z "$garbage_words" ]; then
  steps/cleanup/internal/get_non_scored_words.py $lang > \
    $dir/nonscored_words.txt || exit 1
  
  oov_word=$(cat $lang/oov.txt) || exit 1
  cat $dir/nonscored_words.txt | \
    grep -v -w $oov_word > $dir/garbage_words.txt
else 
  cp $garbage_words $dir/garbage_words.txt || exit 1
fi

if [ ! -z "$silence_words" ]; then
  cp $silence_words $dir/silence_words.txt
fi

echo "<eps>" >> $dir/silence_words.txt

nj=$(cat $ali_dir/num_jobs) || exit 1

$cmd JOB=1:$nj $dir/log/get_arc_info.JOB.log \
  lattice-align-words $lang/phones/word_boundary.int \
    $srcdir/final.mdl \
    "ark:gunzip -c $ali_dir/lat.JOB.gz |" ark:- \| \
  lattice-arc-post --acoustic-scale=$acwt \
    $srcdir/final.mdl ark:- - \| \
  utils/int2sym.pl -f 5 $lang/words.txt \| \
  utils/int2sym.pl -f 6- $lang/phones.txt '>' \
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
    --silence-words=$dir/silence_words.txt \
    --garbage-words=$dir/garbage_words.txt \
    --max-phone-length=$max_phone_len \
    $dir/arc_info_sym.JOB.txt - \| \
  copy-feats ark,t:- \
    ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

for n in $(seq $nj); do
  cat $dir/targets.$n.scp
done > $dir/targets.scp

steps/segmentation/validate_targets_dir.sh $dir $data || exit 1

echo "$0: Done creating targets in $dir/targets.scp"

