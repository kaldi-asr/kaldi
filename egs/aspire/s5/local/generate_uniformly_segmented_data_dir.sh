#!/usr/bin/env bash

# Copyright Vijayaditya Peddinti, 2016.
# Apache 2.0.
# This script generates uniformly segmented data dir, if the directory
# already has a segments file (e.g. in data/dev_aspire) we create directory
# without segments and then uniformly segment it.
# It also extracts hires mfcc features

set -e
set -x

stage=1
num_jobs=30
overlap=5 # size of the overlap
window=10 # size of the uniform segment

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <data-set> <out-data-set>"
  echo " Options:"
  echo "    --stage (1|2|3)  # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet3/tdnn"
  exit 1;
fi

data_set=$1
segmented_data_set=$2

if [ "$data_set" == "dev_aspire" ]; then
  if [ $stage -le 1 ]; then
    echo "$0: Creating the data dir with whole recordings without segmentation"
    # create a whole directory without the segments
    unseg_dir=data/${data_set}_whole_hires
    src_dir=data/${data_set}
    utils/data/convert_data_dir_to_whole.sh $src_dir $unseg_dir

    echo "$0: Creating the $unseg_dir/reco2file_and_channel file"
    cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
  fi
  data_set=${data_set}_whole
else
  utils/copy_data_dir.sh data/$data_set data/${data_set}_hires
fi

if [ $stage -le 2 ]; then
  echo "$0: Extracting features"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $num_jobs \
    --mfcc-config conf/mfcc_hires.conf data/${data_set}_hires

  steps/compute_cmvn_stats.sh data/${data_set}_hires

  utils/fix_data_dir.sh data/${data_set}_hires
  utils/validate_data_dir.sh --no-text data/${data_set}_hires
fi

if [ $stage -le 3 ]; then
  echo "$0: Generating uniform segments with length $window and overlap $overlap."
  [ -d data/${segmented_data_set}_hires ] && rm -r data/${segmented_data_set}_hires
  if [ ! -f data/${data_set}_hires/segments ]; then
    utils/data/get_segments_for_data.sh data/${data_set}_hires > \
      data/${data_set}_hires/segments.tmp
    mv data/${data_set}_hires/segments.tmp data/${data_set}_hires/segments
  fi

  mkdir -p data/${segmented_data_set}_hires

  utils/data/get_uniform_subsegments.py \
    --max-segment-duration=$window \
    --overlap-duration=$overlap \
    --max-remaining-duration=$(perl -e "print $window/ 2.0") \
    data/${data_set}_hires/segments > data/${segmented_data_set}_hires/sub_segments

  utils/data/subsegment_data_dir.sh data/${data_set}_hires \
    data/${segmented_data_set}_hires/sub_segments data/${segmented_data_set}_hires
  steps/compute_cmvn_stats.sh data/${segmented_data_set}_hires

  utils/fix_data_dir.sh data/${segmented_data_set}_hires
  utils/validate_data_dir.sh --no-text data/${segmented_data_set}_hires
fi
