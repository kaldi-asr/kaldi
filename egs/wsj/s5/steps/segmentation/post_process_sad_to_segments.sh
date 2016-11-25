#! /bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e -o pipefail -u
. path.sh

cmd=run.pl
stage=-10

segmentation_config=conf/segmentation.conf
nj=18

frame_shift=0.01
weight_threshold=0.5
ali_suffix=_acwt0.1

phone_map=

. utils/parse_options.sh

if [ $# -ne 5 ] && [ $# -ne 4 ]; then
  echo "This script converts an alignment directory containing per-frame SAD "
  echo "labels or per-frame speech probabilities into kaldi-style "
  echo "segmented data directory. "
  echo "This script first converts the per-frame labels or weights into "
  echo "segmentation and then calls "
  echo "steps/segmentation/internal/post_process_sad_to_segments.sh, "
  echo "which does the actual post-processing step."
  echo "Usage: $0 <data-dir> (<lang> <vad-dir>|<weights-scp>) <segmentation-dir> <segmented-data-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire data/dev_aspire_seg"
  exit 1
fi

data_dir=$1
vad_dir=

if [ $# -eq 5 ]; then
  lang=$2
  vad_dir=$3
  shift; shift; shift
else 
  weights_scp=$2
  shift; shift
fi

dir=$1
segmented_data_dir=$2

cat $data_dir/segments | awk '{print $1" "$2}' | \
  utils/utt2spk_to_spk2utt.pl > $data_dir/reco2utt

utils/split_data.sh $data_dir $nj

for n in `seq $nj`; do
  cat $data_dir/split$nj/$n/segments | awk '{print $1" "$2}' | \
    utils/utt2spk_to_spk2utt.pl > $data_dir/split$nj/$n/reco2utt
done


mkdir -p $dir

if [ ! -z "$vad_dir" ]; then
  nj=`cat $vad_dir/num_jobs` || exit 1

  if [ -z "$phone_map" ]; then
    phone_map=$dir/phone_map

    {
    cat $lang/phones/silence.int | awk '{print $1" 0"}';
    cat $lang/phones/nonsilence.int | awk '{print $1" 1"}';
    } | sort -k1,1 -n > $dir/phone_map
  fi
  
  if [ $stage -le 0 ]; then
    # Convert the original SAD into segmentation
    $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
      segmentation-init-from-ali --reco2utt-rspecifier="ark,t:$data_dir/split$nj/JOB/reco2utt" \
      --segmentation-rspecifier="ark:segmentation-init-from-segments --shift-to-zero=false --frame-shift=$frame_shift $data_dir/split$nj/JOB/segments ark:- |" \
      "ark:gunzip -c $vad_dir/ali${ali_suffix}.JOB.gz |" ark:- \| \
      segmentation-copy --label-map=$phone_map ark:- \
      "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
  fi
else
  for n in `seq $nj`; do
    utils/filter_scp.pl $data_dir/split$nj/$n/reco2utt $weights_scp  > $dir/weights.$n.scp
  done

  $cmd JOB=1:$nj $dir/log/weights_to_segments.JOB.log \
    copy-vector scp:$dir/weights.JOB.scp ark,t:- \| \
    awk -v t=$weight_threshold '{printf $1; for (i=3; i < NF; i++) { if ($i >= t) printf (" 1"); else printf (" 0"); }; print "";}' \| \
    segmentation-init-from-ali --reco2utt-rspecifier="ark,t:$data_dir/split$nj/JOB/reco2utt" \
    --segmentation-rspecifier="ark:segmentation-init-from-segments --shift-to-zero=false --frame-shift=$frame_shift $data_dir/split$nj/JOB/segments ark:- |" \
    ark,t:- "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
fi

echo $nj > $dir/num_jobs

steps/segmentation/internal/post_process_segments.sh \
  --stage $stage --cmd "$cmd" \
  --config $segmentation_config --frame-shift $frame_shift \
  $data_dir $dir $segmented_data_dir
