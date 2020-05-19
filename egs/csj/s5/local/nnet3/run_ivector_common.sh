#!/usr/bin/env bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh
# (and may eventually be called by more scripts). It contains the common feature
# preparation and ivector-related parts of the script. See those scripts for 
# examples of usage.

stage=0
train_set=train_nodup
dev_set=
test_sets="eval1 eval2 eval3"
gmm=tri4

nnet3_affix=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ -e data/train_dev ] ;then
  dev_set=train_dev
fi

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to perturb
  # the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC featuresfor low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1;
fi

if [ $stage -le 3 ]; then
  #Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b1{5,6,7,8}/$USER/kaldi-data/egs/csj-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp $dev_set ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;

  # generate high-resolution MFCC feautres
  for datadir in ${train_set}_sp $dev_set ${test_sets}; do
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: train the diagonal UBM."
  # Previously, the "train_nodup_hires" dataset is used to train the diag_ubm, 
  # the volume is about 1/3 of the "train_nodup_sp_hires", so I use about 1/3 of data.
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/3]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
    $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=4 --right-context=4" \
    --max-utts 10000 --subsample 2 \
    ${temp_data_root}/${train_set}_sp_hires_subset \
    exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 50 \
    --num-frames 500000 --num-threads 8 \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  # Train the iVector extractor.Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data. The script defaults to an iVector dimension of 100
  # even though $nj is just 10, each job uses multiple processes and threads.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 50 \
    data/${train_set}_sp_hires exp/nnet3${nnet3_affix}/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on. With --utts-per-spk-max 2,
  # the scripts pairs the utterances into twos, and treats each of these pairs
  # as one speaker; this gives more diversity in iVectors.
  # Note that these are extracted 'online'.

  # Note, we don't encode the 'max2' in the name of the iVectordir even though
  # that's the data we extract the iVectors from, as it's still going to be 
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  if [[ $(hostname -f ) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/csj-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 50 \
    ${temp_data_root}/${train_set}_sp_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for datadir in $dev_set $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
      data/${datadir}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${datadir}_hires
  done
fi

exit 0;
