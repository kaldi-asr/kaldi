#!/bin/bash

. ./cmd.sh

set -e
stage=1
train_stage=-10
generate_alignments=false
multi_style=false
noise_list="reverb:music:noise:babble:clean"
augment_test_set=false
suffix=""
clean_ali=tri4_ali_nodup
train_set=train_nodup

. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3

if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

if [ "$multi_style" == "true" ]; then
  suffix=_ms
fi

if [ "$multi_style" == "true" ]; then
  if [ $stage -le 1 ]; then
    for x in $train_set ; do
      if [ ! -d "RIRS_NOISES" ]; then
        # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
        wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        unzip rirs_noises.zip
      fi

      if [ ! -f data/$x/reco2dur ]; then
        utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/$x || exit 1;
      fi

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

      # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --prefix "reverb" \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 8000 \
        data/$x data/${x}_reverb

      # Prepare the MUSAN corpus, which consists of music, speech, and noise
      # suitable for augmentation.
      local/make_musan.sh /export/corpora/JHU/musan data

      # Get the duration of the MUSAN recordings.  This will be used by the
      # script augment_data_dir.py.
      for name in speech noise music; do
        utils/data/get_reco2dur.sh data/musan_${name}
      done

      # Augment with musan_noise
      steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spkr-id "true" \
        --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
        data/${x} data/${x}_noise

      # Augment with musan_music
      steps/data/augment_data_dir.py --utt-prefix "music" --modify-spkr-id "true" \
        --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
        data/${x} data/${x}_music

      # Augment with musan_speech
      steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spkr-id "true" \
        --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" \
        data/${x} data/${x}_babble

      # Combine all the noise dirs (This part can be simplified once we know what noise types we will add)
      combine_str=""
      noise_list_parsed=`echo $noise_list | awk -F ":" '{for (i=1; i<=NF; i++) printf "%s ", $i}'`
      for n in $noise_list_parsed; do
        if [ "$n" == "clean" ]; then
          combine_str+="data/$x "
        else
          combine_str+="data/${x}_${n} "
        fi
      done
      utils/combine_data.sh data/${x}_ms $combine_str
    done
  fi

  if [ $stage -le 2 ] && $generate_alignments; then
    # obtain the alignment of augmented data from clean data
    local/copy_ali_dir.sh --nj 40 --cmd "$train_cmd" \
      data/${train_set}_ms exp/${clean_ali} exp/${clean_ali}_ms
  fi
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/swbd-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri2_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set$suffix train_100k_nodup; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

  for dataset in eval2000 train_dev $maybe_rt03; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done

  # Take the first 30k utterances (about 1/8th of the data) this will be used
  # for the diagubm training.
  if [ "$multi_style" == "true" ]; then
    # But since we train i-vector extractor on clean data we filter the clean data
    utils/copy_data_dir.sh data/${train_set}${suffix}_hires data/${train_set}_hires
    utils/filter_scp.pl data/${train_set}/utt2spk data/${train_set}${suffix}_hires/utt2spk > data/${train_set}_hires/utt2spk
    utils/fix_data_dir.sh data/${train_set}_hires || exit 1;
  fi
  utils/subset_data_dir.sh --first data/${train_set}_hires 30000 data/${train_set}_30k_hires
  utils/data/remove_dup_utts.sh 200 data/${train_set}_30k_hires data/${train_set}_30k_nodup_hires  # 33hr
fi

# ivector extractor training
if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/train_100k_nodup_hires \
    data/lang exp/tri2_ali_100k_nodup exp/nnet3/tri3b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_30k_nodup_hires 512 exp/nnet3/tri3b exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_100k_nodup_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 data/${train_set}${suffix}_hires data/${train_set}${suffix}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}${suffix}_max2_hires exp/nnet3/extractor exp/nnet3/ivectors_${train_set}${suffix} || exit 1;

  for data_set in eval2000 train_dev $maybe_rt03; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires exp/nnet3/extractor exp/nnet3/ivectors_$data_set || exit 1;
  done

  if [ "$augment_test_set" == "true" ]; then
    test_set="eval2000 $maybe_rt03"
    for data_set in $test_set; do
      if [ ! -f data/$data_set/reco2dur ]; then
        utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/$data_set || exit 1;
      fi

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
      rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

      # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --prefix "reverb" \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 8000 \
        data/$data_set data/${data_set}_reverb

      # Augment with musan_noise
      steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spkr-id "true" \
        --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
        data/${data_set} data/${data_set}_noise

      # Augment with musan_music
      steps/data/augment_data_dir.py --utt-prefix "music" --modify-spkr-id "true" \
        --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
        data/${data_set} data/${data_set}_music

      # Augment with musan_speech
      steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spkr-id "true" \
        --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" \
        data/${data_set} data/${data_set}_babble

      for noise_type in babble music reverb noise; do
        # Create MFCCs for the eval set
        utils/copy_data_dir.sh data/${data_set}_${noise_type} data/${data_set}_${noise_type}_hires
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
          data/${data_set}_${noise_type}_hires exp/make_hires/${data_set}_${noise_type} $mfccdir;
        steps/compute_cmvn_stats.sh data/${data_set}_${noise_type}_hires exp/make_hires/${data_set}_${noise_type} $mfccdir;
        utils/fix_data_dir.sh data/${data_set}_${noise_type}_hires  # remove segments with problems

        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
            data/${data_set}_${noise_type}_hires exp/nnet3/extractor exp/nnet3/ivectors_${data_set}_${noise_type} || exit 1;
      done
    done
  fi
fi
