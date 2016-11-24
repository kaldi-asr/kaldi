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
    segmentation-copy --label-map=$phone2sad_map ark:- \
    "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
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

mv $segmented_data_dir/segments $segmented_data_dir/sub_segments
utils/data/subsegment_data_dir.sh $data_dir $segmented_data_dir/sub_segments $segmented_data_dir

utils/utt2spk_to_spk2utt.pl $segmented_data_dir/utt2spk > $segmented_data_dir/spk2utt || exit 1
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

