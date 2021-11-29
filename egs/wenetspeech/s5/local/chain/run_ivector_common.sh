#!/usr/bin/env bash
# Copyright 2021 Xiaomi Corporation (Author: Yongqing Wang)
#           2021 ASLP, NWPU (Author: Hang Lyu)
# Apache 2.0

# This script is copied from egs/gigaspeech/s5/local/chain/run_ivector_common.sh
# and modified.

set -e -o pipefail

# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
train_nj=50
train_set=train    # you might set this to e.g. train
test_sets=""
gmm=tri4b_cleaned         # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.
num_threads_ubm=16
num_processes=4
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
do_speedperturb=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  if $do_speedperturb; then
    # Although the nnet will be trained by high resolution data, we still have to
    # perturb the normal data to get the alignment.  _sp stands for speed-perturbed
    echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
    utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
    echo "$0: making MFCC features for low-resolution speed-perturbed data"
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $train_nj data/${train_set}_sp || exit 1;
    steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
    echo "$0: fixing input data-dir to remove nonexistent features, in case some "
    echo ".. speed-perturbed segments were too short."
    utils/fix_data_dir.sh data/${train_set}_sp
  else
    ln -sf $train_set data/${train_set}_sp
  fi
fi

if [ $stage -le 2 ]; then
  echo -e "======Align all data START|current time : `date +%Y-%m-%d-%T`======"
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --stage 0 --nj $train_nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
  echo -e "======Align all data END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 3 ]; then
  echo -e "======High-mfcc START|current time : `date +%Y-%m-%d-%T`======"
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data

  for part in ${train_set}_sp $test_sets; do
    utils/copy_data_dir.sh data/$part data/${part}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.

  # By default, we skip the volume perturbation in WeNetSpeech.
  #utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for part in ${train_set}_sp $test_sets; do
    steps/make_mfcc.sh --nj $train_nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${part}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_hires || exit 1;
    utils/fix_data_dir.sh data/${part}_hires
  done

  echo -e "======High-mfcc END|current time : `date +%Y-%m-%d-%T`======"
fi


if [ $stage -le 4 ]; then
  echo "$0: making a subset of data to train the diagonal UBM and the PCA transform."
  # We'll one 50th of the data, since Librispeech is very large.
  mkdir -p exp/${train_set}/nnet3${nnet3_affix}/diag_ubm

  total_num=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  subset_num=$((total_num/50))
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
     $subset_num data/${train_set}_sp_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       data/${train_set}_sp_hires_subset \
       exp/${train_set}/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $train_nj \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    data/${train_set}_sp_hires_subset 512 \
    exp/${train_set}/nnet3${nnet3_affix}/pca_transform \
    exp/${train_set}/nnet3${nnet3_affix}/diag_ubm
fi


if [ $stage -le 5 ]; then
  echo -e "======Ivector extractor START|current time : `date +%Y-%m-%d-%T`======"
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 1d50 subset (about one fifth of the data, or 200 hours).
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 50 \
    --num-processes $num_processes \
    data/${train_set}_sp_hires_subset \
    exp/${train_set}/nnet3${nnet3_affix}/diag_ubm \
    exp/${train_set}/nnet3${nnet3_affix}/extractor || exit 1;
  echo -e "======Ivector extractor END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 6 ]; then
  echo -e "======Extractor train START|current time : `date +%Y-%m-%d-%T`======"
  echo "$0: extracting iVectors for training data"
  ivectordir=exp/${train_set}/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker. this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${ivectordir}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
    --nj $train_nj \
    ${ivectordir}/${train_set}_sp_hires_max2 \
    exp/${train_set}/nnet3${nnet3_affix}/extractor \
    $ivectordir || exit 1;
  echo -e "======Extractor train END|current time : `date +%Y-%m-%d-%T`======"
fi

if [ $stage -le 7 ]; then
  echo "$0: extracting iVectors for dev and test data"
  for part in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj $train_nj \
      data/${part}_hires exp/${train_set}/nnet3${nnet3_affix}/extractor \
      exp/${train_set}/nnet3${nnet3_affix}/ivectors_${part}_hires || exit 1;
  done
fi

exit 0;
