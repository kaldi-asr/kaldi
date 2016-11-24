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

frame_subsampling_factor=1

phone2sad_map=

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

utils/data/get_reco2utt.sh $data_dir

mkdir -p $dir

if [ ! -z "$vad_dir" ]; then
  nj=`cat $vad_dir/num_jobs` || exit 1
  
  utils/split_data.sh $data_dir $nj

  for n in `seq $nj`; do
    cat $data_dir/split$nj/$n/segments | awk '{print $1" "$2}' | \
      utils/utt2spk_to_spk2utt.pl > $data_dir/split$nj/$n/reco2utt
  done

  if [ -z "$phone2sad_map" ]; then
    phone2sad_map=$dir/phone2sad_map

    {
    cat $lang/phones/silence.int | awk '{print $1" 0"}';
    cat $lang/phones/nonsilence.int | awk '{print $1" 1"}';
    } | sort -k1,1 -n > $dir/phone2sad_map
  fi
  
  frame_shift_subsampled=`perl -e "print ($frame_subsampling_factor * $frame_shift)"`

  if [ $stage -le 0 ]; then
    # Convert the original SAD into segmentation
    $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
      segmentation-init-from-ali \
      "ark:gunzip -c $vad_dir/ali${ali_suffix}.JOB.gz |" ark:- \| \
      segmentation-combine-segments ark:- \
      "ark:segmentation-init-from-segments --shift-to-zero=false --frame-shift=$frame_shift_subsampled $data_dir/split$nj/JOB/segments ark:- |" \
      "ark,t:$data_dir/split$nj/JOB/reco2utt" ark:- \| \
      segmentation-copy --label-map=$phone2sad_map \
      --frame-subsampling-factor=$frame_subsampling_factor ark:- \
      "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
  fi
else
  utils/split_data.sh $data_dir $nj

  for n in `seq $nj`; do
    utils/data/get_reco2utt.sh $data_dir/split$nj/$n
    utils/filter_scp.pl $data_dir/split$nj/$n/reco2utt $weights_scp > \
      $dir/weights.$n.scp
  done

  $cmd JOB=1:$nj $dir/log/weights_to_segments.JOB.log \
    copy-vector scp:$dir/weights.JOB.scp ark,t:- \| \
    awk -v t=$weight_threshold '{printf $1; for (i=3; i < NF; i++) { if ($i >= t) printf (" 1"); else printf (" 0"); }; print "";}' \| \
    segmentation-init-from-ali \
    ark,t:- ark:- \| segmentation-combine-segments ark:- \
    "ark:segmentation-init-from-segments --shift-to-zero=false --frame-shift=$frame_shift_subsampled $data_dir/split$nj/JOB/segments ark:- |" \
    "ark,t:$data_dir/split$nj/JOB/reco2utt" ark:- \| \
    segmentation-copy --frame-subsampling-factor=$frame_subsampling_factor \
    ark:- "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
fi

echo $nj > $dir/num_jobs

if [ $stage -le 1 ]; then
  rm -r $segmented_data_dir || true
  utils/data/convert_data_dir_to_whole.sh $data_dir $segmented_data_dir || exit 1
  rm $segmented_data_dir/text || true
fi

steps/segmentation/internal/post_process_segments.sh \
  --stage $stage --cmd "$cmd" \
  --config $segmentation_config --frame-shift $frame_shift \
  $data_dir $dir $segmented_data_dir

utils/utt2spk_to_spk2utt.pl $segmented_data_dir/utt2spk > $segmented_data_dir/spk2utt || exit 1
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

