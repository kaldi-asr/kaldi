#!/bin/bash

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

if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <data-set>"
  echo " Options:"
  echo "    --stage (1|2|3)  # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet3/tdnn"
  exit 1;
fi

data_set=$1

if [ "$data_set" == "dev_aspire" ]; then
  if [ $stage -le 1 ]; then
    echo "$0 : Creating the data dir with whole recordings without segmentation"
    # create a whole directory without the segments
    unseg_dir=data/${data_set}_whole
    src_dir=data/$data_set
    mkdir -p $unseg_dir
    echo "$0 : Creating the $unseg_dir/wav.scp file"
    cp $src_dir/wav.scp $unseg_dir

    echo "$0 : Creating the $unseg_dir/reco2file_and_channel file"
    cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
    cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt

  fi
  data_set=${data_set}_whole
fi

segmented_data_set=${data_set}_uniformsegmented_win${window}_over${overlap}
if [ $stage -le 2 ]; then
  echo "$0 : Generating uniform segments with length $window and overlap $overlap."
  [ -d data/$segmented_data_set ] && rm -r data/$segmented_data_set
  utils/copy_data_dir.sh --validate-opts "--no-text" \
    data/$data_set data/$segmented_data_set
  cp data/$data_set/reco2file_and_channel data/$segmented_data_set

  local/multi_condition/create_uniform_segments.py \
    --overlap $overlap --window $window data/$segmented_data_set

  for file in cmvn.scp feats.scp; do
    rm -f data/$segmented_data_set/$file
  done
  utils/validate_data_dir.sh --no-text --no-feats data/$segmented_data_set
fi

if [ $stage -le 3 ]; then
  echo "$0 : Extracting features for the uniformly segmented dir"
  [ -d data/${segmented_data_set}_hires ] && rm -r data/${segmented_data_set}_hires
  utils/copy_data_dir.sh --validate-opts "--no-text " \
    data/${segmented_data_set} data/${segmented_data_set}_hires

  steps/make_mfcc.sh --cmd "$train_cmd" --nj $num_jobs \
    --mfcc-config conf/mfcc_hires.conf data/${segmented_data_set}_hires

  steps/compute_cmvn_stats.sh data/${segmented_data_set}_hires

  utils/fix_data_dir.sh data/${segmented_data_set}_hires
  utils/validate_data_dir.sh --no-text data/${segmented_data_set}_hires
fi
