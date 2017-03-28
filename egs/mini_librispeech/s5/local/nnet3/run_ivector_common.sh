#!/bin/bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
min_seg_len=1.55
train_set=train_clean_5
gmm=tri3b

nnet3_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: combining short segments of low-resolution speed-perturbed MFCC data"
  src=data/${train_set}_sp
  dest=data/${train_set}_sp_comb
  utils/data/combine_short_segments.sh $src $min_seg_len $dest || exit 1
  # re-use the CMVN stats from the source directory, since it seems to be slow to
  # re-compute them after concatenating short segments.
  cp $src/cmvn.scp $dest/
  utils/fix_data_dir.sh $dest

  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed, short-segment-combined low-resolution data"
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    $dest data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b1{5,6,7,8}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev_clean_2; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;

  for datadir in ${train_set}_sp dev_clean_2; do
    steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  src=data/${train_set}_sp_hires
  dest=data/${train_set}_sp_hires_comb
  utils/data/combine_short_segments.sh $src $min_seg_len $dest || exit 1;
  # just copy over the CMVN to avoid having to recompute it.
  cp $src/cmvn.scp $dest/
  utils/fix_data_dir.sh $dest
fi

# if [ $stage -le 5 ]; then
#   # We need to build a small system just because we need the LDA+MLLT transform
#   # to train the diag-UBM on top of.  We align a subset of training data for
#   # this purpose.
#   echo "$0: aligning a subset of training data."
#   steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
#     data/${train_set}_sp data/lang exp/${gmm} exp/${gmm}_ali_${train_set}_sp || exit 1;
# fi

if [ $stage -le 6 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
    echo "$0: training a system on the hires subset data for its LDA+MLLT transform, in order to produce the diagonal GMM."
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/${train_set}_sp_hires_comb data/lang \
    $ali_dir exp/nnet3${nnet3_affix}/tri4b || exit 1
fi
