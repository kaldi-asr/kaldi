#!/bin/bash
# Copyright 2020 Xuechen LIU
# Apache 2.0.
#
# See ../README.md for more info on data required.
# Results (equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/exp/mfcc
vaddir=`pwd`/exp/mfcc

nj=40
stage=-10
include_aishell2=false # by default we do not have AISHELL data involved since
                       # it is not available immediately for everybody

corporadir=corpora/    # this path is for storing downloaded corpora
nnetdir=exp/xvector_nnet_1a

aishell2_root=/media/hdd3/AISHELL2
himia_root=corpora/himia_corpus
musan_root=/media/hdd3/musan

. ./utils/parse_options.sh

if [ $stage -le -10 ]; then
    # prepare DNN training data. Note that here we provide options for users,
    # in case they do not have access to AISHELL2 data.
    # OpenSLR side is basically a pseudo-copy from multi_cn solutions.
    local/prepare_multi_cn.sh --stage 0 corpora/openslr || exit 1;
fi

if [ $stage -le -1 ]; then
    if $include_aishell2; then
        # check if AISHELL2 corpus exists
        [ -d $aishell2_root/iOS/data ] || (echo $aishell2_root does not exist && exit 1;)
        # AISHELL2 preparation
        local/prepare_aishell2.sh $aishell2_root/iOS/data \
            data/aishell2/local/train data/aishell2/train
        utils/fix_data_dir.sh data/aishell2/train || exit 1;
        utils/combine_data.sh data/train \
            data/aishell2/train data/train_combined || exit 1;
    else
        mv data/train_combined data/train
    fi
fi

if [ $stage -le 0 ]; then
    # prepare HIMIA. note that since training of neural net is done
    # via single-channel data, we here by dafault perform data processing
    # as 'single channel' as well. (but we do test on multi-channel...as well)
    for set in test_v2 dev train; do
        local/download_and_untar.sh $himia_root http://www.openslr.org/resources/85/$set.tar.gz $set || exit 1;
        [[ "$set" == "test_v2" ]] && set=test
            python local/himia_data_prep.py $himia_root/$set data/himia/$set || exit 1;
            utils/utt2spk_to_spk2utt.pl data/himia/$set/utt2spk > data/himia/$set/spk2utt
            utils/fix_data_dir.sh data/himia/$set
    done

    # This is a simple filtering operation for both trial script, in order
    # to provide workable trial list
    python local/himia_trials_prep.py $himia_root/test/trials_1m \
        $himia_root/test/wav.scp data/himia/test/trials_1m_full 'sc' || exit 1;
    python local/himia_trials_prep.py $himia_root/test/trials_mic \
        $himia_root/test/wav.scp data/himia/test/trials_mic_full 'mc' || exit 1;
fi

if [ $stage -le 1 ]; then
    for set in train himia/test himia/dev himia/train; do
        steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
            data/$set exp/make_mfccs/$set/log exp/make_mfccs/$set || exit 1;
        utils/fix_data_dir.sh data/$set
        sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
            data/$set exp/make_vad/$set/log exp/make_vad/$set
        steps/compute_cmvn_stats.sh data/$set || exit 1;
        utils/fix_data_dir.sh data/$set
    done
fi
# data augmentation
if [ $stage -le 2 ]; then
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

    if [ ! -d "RIRS_NOISES" ]; then
        # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
        wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
        unzip rirs_noises.zip
    fi

    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
    # additive noise here.
    steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/train data/train_reverb
    cp data/train/vad.scp data/train_reverb/
    utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
    rm -rf data/train_reverb
    mv data/train_reverb.new data/train_reverb

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done
    # Augment with musan_noise
    steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
    # Augment with musan_music
    steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
    # Augment with musan_speech
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

    # Combine reverb, noise, music, and babble into one directory.
    utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi

if [ $stage -le 3 ]; then
    # Take a random subset of the augmentations
    utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
    utils/fix_data_dir.sh data/train_aug_1m

    # Make MFCCs for the augmented data.  Note that we do not compute a new
    # vad.scp file here.  Instead, we use the vad.scp from the clean version of
    # the list.
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
        data/train_aug_1m exp/make_mfccs/train_aug_1m/log exp/make_mfccs/train_aug_1m

    # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
    # double the size of the original clean list.
    utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi

if [ $stage -le 4 ]; then
    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "$train_cmd" \
        data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
    utils/fix_data_dir.sh data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 4s (400 frames) per utterance.
    min_len=400
    mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
    utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
    mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
    utils/fix_data_dir.sh data/train_combined_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    min_num_utts=8
    awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
    awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
    mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
    utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk

    utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
    mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames

    # Now we're ready to create training examples.
    utils/fix_data_dir.sh data/train_combined_no_sil
fi

if [ $stage -le 6 ]; then
    # Stages 6 through 8 are handled in run_xvector.sh
    local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
        --data data/train_combined_no_sil --nnet-dir $nnetdir \
        --egs-dir $nnetdir/egs
fi

if [ $stage -le 20 ]; then
    # The HIMIA challenge has text-dependent track but in order to provide
    # a flexible repository, we only implemented text-independent part for now
    local/run_text_independent.sh $nnetdir data/himia/test/trials_1m_full \
        data/himia/test/trials_mic_full || exit 1;
fi
