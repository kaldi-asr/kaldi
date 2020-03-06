#!/usr/bin/env bash
# Copyright 2019   Phani Sankar Nidadavolu
# Apache 2.0.

. ./cmd.sh

set -e
stage=0
aug_list="reverb music noise babble clean"  #clean refers to the original train dir
use_ivectors=true
num_reverb_copies=1

# Alignment directories
lda_mllt_ali=tri2_ali_100k_nodup
clean_ali=tri4_ali_nodup

# train directories for ivectors and TDNNs
ivector_trainset=train_100k_nodup
train_set=train_nodup

. ./path.sh
. ./utils/parse_options.sh

if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

if [ $stage -le 0 ]; then
  # Adding simulated RIRs to the original data directory
  echo "$0: Preparing data/${train_set}_reverb directory"

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ ! -f data/$train_set/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/$train_set || exit 1;
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD train_nodup.
  # Note that we don't add any additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_reverb_copies \
    --source-sampling-rate 8000 \
    data/$train_set data/${train_set}_reverb
fi

if [ $stage -le 1 ]; then
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # We will use them as additive noises for data augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 --use-vocals "true" \
        /export/corpora/JHU/musan data

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
    --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
    data/${train_set} data/${train_set}_noise

  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id "true" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
    data/${train_set} data/${train_set}_music

  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id "true" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/musan_speech" \
    data/${train_set} data/${train_set}_babble

  # Combine all the augmentation dirs
  # This part can be simplified once we know what noise types we will add
  combine_str=""
  for n in $aug_list; do
    if [ "$n" == "clean" ]; then
      # clean refers to original of training directory
      combine_str+="data/$train_set "
    else
      combine_str+="data/${train_set}_${n} "
    fi
  done
  utils/combine_data.sh data/${train_set}_aug $combine_str
fi

if [ $stage -le 2 ]; then
  # Extract low-resolution MFCCs for the augmented data
  # To be used later to generate alignments for augmented data
  echo "$0: Extracting low-resolution MFCCs for the augmented data. Useful for generating alignments"
  mfccdir=mfcc_aug
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/swbd-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
                     data/${train_set}_aug exp/make_mfcc/${train_set}_aug $mfccdir
  steps/compute_cmvn_stats.sh data/${train_set}_aug exp/make_mfcc/${train_set}_aug $mfccdir
  utils/fix_data_dir.sh data/${train_set}_aug || exit 1;
fi

if [ $stage -le 3 ] && $generate_alignments; then
  # obtain the alignment of augmented data from clean data
  include_original=false
  prefixes=""
  for n in $aug_list; do
    if [ "$n" == "reverb" ]; then
      for i in `seq 1 $num_reverb_copies`; do
        prefixes="$prefixes "reverb$i
      done
    elif [ "$n" != "clean" ]; then
      prefixes="$prefixes "$n
    else
      # The original train directory will not have any prefix
      # include_original flag will take care of copying the original alignments
      include_original=true
    fi
  done
  echo "$0: Creating alignments of aug data by copying alignments of clean data"
  steps/copy_ali_dir.sh --nj 40 --cmd "$train_cmd" \
    --include-original "$include_original" --prefixes "$prefixes" \
    data/${train_set}_aug exp/${clean_ali} exp/${clean_ali}_aug
fi

if [ $stage -le 4 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/swbd-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in ${train_set}_aug; do
    echo "$0: Creating hi resolution MFCCs for dir data/$dataset"
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi

if [ $stage -le 5 ]; then
  mfccdir=mfcc_hires
  for dataset in eval2000 train_dev $maybe_rt03; do
    echo "$0: Creating hi resolution MFCCs for data/$dataset"
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi

if [ "$use_ivectors" == "true" ]; then
  if [ $stage -le 6 ]; then
    # Take  30k utterances from MS data this will be used for the diagubm training.
    utils/subset_data_dir.sh data/${train_set}_aug_hires 30000 data/${train_set}_aug_30k_hires
    utils/data/remove_dup_utts.sh 200 data/${train_set}_aug_30k_hires data/${train_set}_aug_30k_nodup_hires  # 33hr

    # Make a 140 hr subset of augmented data to train i-vector extractor
    # we don't extract hi res features again for ivector training data
    # we take it from the ms features extracted on the entire training set
    # First augment the train_100k_nodup directory which is used to train the i-vector extractor in baseline
    utils/copy_data_dir.sh data/${train_set}_aug_hires data/${ivector_trainset}_aug_hires
    utils/filter_scp.pl -f 2 data/${ivector_trainset}/utt2spk data/${train_set}_aug_hires/utt2uniq | \
        utils/filter_scp.pl - data/${train_set}_aug_hires/utt2spk > data/${ivector_trainset}_aug_hires/utt2spk
    utils/fix_data_dir.sh data/${ivector_trainset}_aug_hires

    # Since the data size is now increased make a subset of it to bring the duration back to required size (140hr)
    utils/subset_data_dir.sh data/${ivector_trainset}_aug_hires 100000 data/${ivector_trainset}_aug_hires_subset
    utils/data/remove_dup_utts.sh 200 data/${ivector_trainset}_aug_hires_subset data/${ivector_trainset}_aug_hires
    steps/compute_cmvn_stats.sh data/${ivector_trainset}_aug_hires exp/make_hires/${ivector_trainset} $mfccdir;
    utils/fix_data_dir.sh data/${ivector_trainset}_aug_hires
  fi

  # ivector extractor training
  if [ $stage -le 7 ]; then
    # First copy the clean alignments to augmented alignments to train LDA+MLLT transform
    # Since the alignments are created using  low-res mfcc features make a copy of ivector training directory
    utils/copy_data_dir.sh data/${ivector_trainset}_aug_hires data/${ivector_trainset}_aug
    utils/filter_scp.pl data/${ivector_trainset}_aug/utt2spk data/${train_set}_aug/feats.scp > data/${ivector_trainset}_aug/feats.scp
    utils/fix_data_dir.sh data/${ivector_trainset}_aug
    echo "$0: Creating alignments of aug data by copying alignments of clean data"
    steps/copy_ali_dir.sh --nj 40 --cmd "$train_cmd" \
        data/${ivector_trainset}_aug exp/${lda_mllt_ali} exp/${lda_mllt_ali}_aug

    # We need to build a small system just because we need the LDA+MLLT transform
    # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
    # the transform (12th iter is the last), any further training is pointless.
    # this decision is based on fisher_english
    steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --splice-opts "--left-context=3 --right-context=3" \
      5500 90000 data/${ivector_trainset}_aug_hires \
      data/lang exp/${lda_mllt_ali}_aug exp/nnet3/tri3b
  fi

  if [ $stage -le 8 ]; then
    # To train a diagonal UBM we don't need very much data, so use the smallest subset.
    echo "$0: Training diagonal UBM for i-vector extractor"
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
      data/${train_set}_aug_30k_nodup_hires 512 exp/nnet3/tri3b exp/nnet3/diag_ubm
  fi

  if [ $stage -le 9 ]; then
    # iVector extractors can be sensitive to the amount of data, but this one has a
    # fairly small dim (defaults to 100) so we don't use all of it, we use just the
    # 100k subset (just under half the data).
    echo "$0: Training i-vector extractor for speaker adaptation"
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      data/${ivector_trainset}_aug_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
  fi

  if [ $stage -le 10 ]; then
    # We extract iVectors on all the train_nodup data, which will be what we
    # train the system on.
    # having a larger number of speakers is helpful for generalization, and to
    # handle per-utterance decoding well (iVector starts at zero).
    echo "$0: Extracting ivectors for train and eval directories"
    utils/data/modify_speaker_info.sh --utts-per-spk-max 2 data/${train_set}_aug_hires data/${train_set}_aug_max2_hires

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${train_set}_aug_max2_hires exp/nnet3/extractor exp/nnet3/ivectors_${train_set}_aug || exit 1;

    for dataset in eval2000 train_dev $maybe_rt03; do
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${dataset}_hires exp/nnet3/extractor exp/nnet3/ivectors_$dataset || exit 1;
    done
  fi
fi
