#!/usr/bin/env bash

# this script contains some common (shared) parts of the run_nnet*.sh scripts.
. cmd.sh


stage=0
gmmdir=exp/tri4
speed_perturb=false
trainset=train_clean

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    echo "$0: preparing directory for speed-perturbed data"
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed
    for datadir in ${trainset} ; do
	  utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp 

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 \
        data/${datadir}_sp exp/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_sp exp/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done
  fi

  if [ $stage -le 2 ]; then
	echo "$0: aligning with the perturbed low-resolution data"
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/${trainset}_sp data/lang_nosp ${gmmdir} ${gmmdir}_ali_${trainset}_sp || exit 1
  fi
  trainset=${trainset}_sp
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.

  echo "$0: creating high-resolution MFCC features"  
  for datadir in ${trainset} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires  || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires  || exit 1;
  done

  # We need to build a small system just because we need PCA transform
  # to train the diag-UBM on top of.  
  utils/subset_data_dir.sh data/${trainset}_hires 30000 data/train_30k_hires
fi


if [ $stage -le 4 ]; then
  # Train a small system just for its PCA transform.  
  echo "$0: computing a PCA transform from the hires data."
  mkdir exp -p exp/nnet3
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 30000 --subsample 2 \
    data/train_30k_hires exp/nnet3/pca_transform
fi

if [ $stage -le 5 ]; then
  # To train a diagonal UBM we don't need very much data, so use a small subset
  echo "$0: training the diagonal UBM."
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
    data/train_30k_hires 512 exp/nnet3/pca_transform exp/nnet3/diag_ubm
fi

if [ $stage -le 6 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of 100
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${trainset}_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  ivectordir=exp/nnet3/ivectors_${trainset}_hires

  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  echo "$0: extracing iVector using trained iVector extractor"
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${trainset}_hires data/${trainset}_hires_max2
  
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/${trainset}_hires_max2 exp/nnet3/extractor $ivectordir || exit 1;
fi


exit 0;
