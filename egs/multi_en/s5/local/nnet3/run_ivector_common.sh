#!/usr/bin/env bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/nnet3/run_ivector_common.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Modified paths to match multi_en naming conventions
###########################################################################################

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing chain training
speed_perturb=true
multi=multi_a
gmm=tri5a

. ./path.sh
. ./utils/parse_options.sh

# perturbed data preparation
train_set=$multi/$gmm
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed
    echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
    utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
    echo "$0: making MFCC features for low-resolution speed-perturbed data" 
    steps/make_mfcc.sh --nj 70 --cmd "$train_cmd" \
      data/${train_set}_sp || exit 1
    steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1
    utils/fix_data_dir.sh data/${train_set}_sp || exit 1
  fi

  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/${train_set}_sp data/lang_${multi}_${gmm} exp/$multi/$gmm exp/$multi/${gmm}_ali_sp || exit 1
  fi
  train_set=$multi/${gmm}_sp
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,3,7,9}/$USER/kaldi-data/mfcc/multi-en-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/${multi}/${gmm}_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set $multi/train_100k_nodup; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # do volume-perturbation on the training data prior to extracting hires
    # features; this helps make trained nnets more invariant to test data volume.
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

  for dataset in eval2000 rt03; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset/test data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done

  # Take the first 30k utterances (about 1/8th of the data) this will be used
  # for the diagubm training
  utils/subset_data_dir.sh --first data/${train_set}_hires 30000 data/${train_set}_30k_hires
  utils/data/remove_dup_utts.sh 200 data/${train_set}_30k_hires data/${train_set}_30k_nodup_hires  # 33hr
fi

if [ $stage -le 5 ]; then
  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    data/${train_set}_30k_nodup_hires exp/$multi/nnet3/pca
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  echo "$0: training the diagonal UBM."
  steps/online/nnet2/train_diag_ubm.sh  --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_30k_nodup_hires 512 exp/$multi/nnet3/pca exp/$multi/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/$multi/train_100k_nodup_hires exp/$multi/nnet3/diag_ubm exp/$multi/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  ivectordir=exp/$multi/nnet3/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,3,7,9}/$USER/kaldi-data/ivectors/fisher_swbd-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi


  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_hires ${temp_data_root}/${train_set}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    ${temp_data_root}/${train_set}_hires_max2 \
    exp/$multi/nnet3/extractor $ivectordir

  # Also extract iVectors for the test data
  for data_set in eval2000 rt03; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires exp/$multi/nnet3/extractor exp/$multi/nnet3/ivectors_${data_set} || exit 1;
  done
fi

exit 0;
