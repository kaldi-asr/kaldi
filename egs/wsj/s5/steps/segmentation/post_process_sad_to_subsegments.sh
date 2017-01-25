#! /bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e -o pipefail -u
. path.sh

cmd=run.pl
stage=-10

segmentation_config=conf/segmentation.conf
nj=18

frame_subsampling_factor=1
frame_shift=0.01
frame_overlap=0.015

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-dir> <phone2sad-map> <vad-dir> <segmentation-dir> <segmented-data-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire data/dev_aspire_seg"
  exit 1
fi

data_dir=$1
phone2sad_map=$2
vad_dir=$3
dir=$4
segmented_data_dir=$5

mkdir -p $dir

nj=`cat $vad_dir/num_jobs` || exit 1
  
utils/split_data.sh $data_dir $nj

if [ $stage -le 0 ]; then
  # Convert the original SAD into segmentation
  $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
    segmentation-init-from-ali \
    "ark:gunzip -c $vad_dir/ali.JOB.gz |" ark:- \| \
    segmentation-copy --label-map=$phone2sad_map \
    --frame-subsampling-factor=$frame_subsampling_factor ark:- \
    "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
fi

echo $nj > $dir/num_jobs

# Create a temporary directory into which we can create the new segments
# file.
if [ $stage -le 1 ]; then
  rm -r $segmented_data_dir || true
  utils/data/convert_data_dir_to_whole.sh $data_dir $segmented_data_dir || exit 1
  rm $segmented_data_dir/text || true
fi

if [ $stage -le 2 ]; then
  # --frame-overlap is set to 0 to not do any additional padding when writing
  # segments. This padding will be done later by the option
  # --segment-end-padding to utils/data/subsegment_data_dir.sh.
  steps/segmentation/internal/post_process_segments.sh \
    --stage $stage --cmd "$cmd" \
    --config $segmentation_config --frame-shift $frame_shift \
    --frame-overlap 0 \
    $data_dir $dir $segmented_data_dir
fi

mv $segmented_data_dir/segments $segmented_data_dir/sub_segments
utils/data/subsegment_data_dir.sh --segment-end-padding `perl -e "print $frame_overlap"` \
  $data_dir $segmented_data_dir/sub_segments $segmented_data_dir
utils/fix_data_dir.sh $segmented_data_dir

utils/data/get_reco2num_frames.sh --nj $nj --cmd "$cmd" ${data_dir} 
mv $segmented_data_dir/feats.scp $segmented_data_dir/feats.scp.tmp
cat $segmented_data_dir/segments | awk '{print $1" "$2}' | \
  utils/apply_map.pl -f 2 $data_dir/reco2num_frames > \
  $segmented_data_dir/utt2max_frames
cat $segmented_data_dir/feats.scp.tmp | \
  utils/data/fix_subsegmented_feats.pl $segmented_data_dir/utt2max_frames > \
  $segmented_data_dir/feats.scp

utils/utt2spk_to_spk2utt.pl $segmented_data_dir/utt2spk > \
  $segmented_data_dir/spk2utt || exit 1
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

