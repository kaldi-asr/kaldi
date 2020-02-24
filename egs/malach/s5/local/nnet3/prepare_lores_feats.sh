#!/usr/bin/env bash

set -e -o pipefail

# this is called from the other nnet and chain training scripts.
# It prepares normal-resolution MFCC features for purposes of getting
# alignments and/or lattices on the speed-perturbed data.
#
# please see local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh for examples of
# usage.

stage=0
nj=30
min_seg_len=1.55  # min length in seconds... we do this because chain training
                  # will discard segments shorter than 1.5 seconds.  Must remain in
                  # sync with the same option given to run_ivector_common.sh.
                  # Set it to empty string to skip combining segments.

train_set=train   # you might set this to e.g. train_cleaned.


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


for f in data/${train_set}/utt2spk; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 9 ]; then
  echo "$0: $feats already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi


if [ $stage -le 8 ]; then
  echo "$0: preparing directory for speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 9 ]; then
  echo "$0: making MFCC features for speed-perturbed data"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ ! -z "$min_seg_len" ]; then
  if [ $stage -le 10 ]; then
    echo "$0: combining short segments of 13-dimensional speed-perturbed MFCC data"
    src=data/${train_set}_sp
    dest=data/${train_set}_sp_comb
    utils/data/combine_short_segments.sh $src $min_seg_len $dest
    # re-use the CMVN stats from the source directory, since it seems to be slow to
    # re-compute them after concatenating short segments.
    cp $src/cmvn.scp $dest/
    utils/fix_data_dir.sh $dest
  fi
fi


exit 0;
