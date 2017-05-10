#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -o pipefail
set -u

. path.sh

cmd=run.pl

. parse_options.sh

if [ $# -ne 3 ]; then
  echo "This script converts the alignment in the alignment directory "
  echo "to speech activity segments based on the provided phone-map."
  echo "The output is stored in sad_seg.*.ark along with an scp-file "
  echo "sad_seg.scp in Segmentation format.\n"
  echo "If alignment directory has frame_subsampling_factor, the segments "
  echo "are applied that frame-subsampling-factor.\n"
  echo "The phone-map file must have two columns: "
  echo "<phone-integer-id> <vad-class-label>\n" 
  echo "\n"
  echo "Usage: $0 <ali-dir> <phone-map> <vad-dir>"
  echo "e.g. : $0 exp/tri3_ali data/lang/phones/sad.map exp/tri3_ali_vad"
  exit 1
fi

ali_dir=$1
phone_map=$2
dir=$3

for f in $phone_map $ali_dir/ali.1.gz $ali_dir/final.mdl; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find $f" && exit 1
  fi
done

mkdir -p $dir

nj=`cat $ali_dir/num_jobs` || exit 1
echo $nj > $dir/num_jobs

frame_subsampling_factor=1
if [ -f $ali_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $ali_dir/frame_subsampling_factor`
fi

dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

$cmd JOB=1:$nj $dir/log/get_sad.JOB.log \
  segmentation-init-from-ali \
    "ark:gunzip -c ${ali_dir}/ali.JOB.gz | ali-to-phones --per-frame ${ali_dir}/final.mdl ark:- ark:- |" \
    ark:- \| \
  segmentation-copy --label-map=$phone_map \
    --frame-subsampling-factor=$frame_subsampling_factor ark:- ark:- \| \
  segmentation-post-process --merge-adjacent-segments ark:- \
    ark,scp:$dir/sad_seg.JOB.ark,$dir/sad_seg.JOB.scp

for n in `seq $nj`; do 
  cat $dir/sad_seg.$n.scp
done | sort -k1,1 > $dir/sad_seg.scp
