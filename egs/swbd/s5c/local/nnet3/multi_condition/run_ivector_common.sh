#!/bin/bash 
#set -e
# this script is based on local/nnet3/run_ivector_common.sh
# but it operates on corrupted training/dev/test data sets

. cmd.sh

stage=1
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_data_reps=1
clean_data_dir=train_nodup
iv_dir=exp/nnet3_rvb
speed_perturb=true

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p $iv_dir

if [ "$speed_perturb" == "true" ]; then
  # perturbed data preparation
  if [ $stage -le 1 ] && [ ! -d data/${clean_data_dir}_sp ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in ${clean_data_dir}; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
      utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
      rm -r data/temp1 data/temp2

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      utils/fix_data_dir.sh data/${datadir}_sp
      rm -r data/temp0 data/${datadir}_tmp
    done
  fi


  if [ $stage -le 2 ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/${clean_data_dir}_sp data/lang_nosp exp/tri4 exp/tri4_ali_nodup_sp || exit 1
  fi

  clean_data_dir=${clean_data_dir}_sp
fi


if [ $stage -le 3 ]; then
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # corrupt the data to generate reverberated data 
  # this script modifies wav.scp to include the reverberation commands, the real computation will be done at the feature extraction
  python steps/data/reverberate_data_dir.py \
    --prefix "rev" \
    --rir-set-parameters "0.25, RIRS_NOISES/simulated_rirs/smallroom/rir_list" \
    --rir-set-parameters "0.25, RIRS_NOISES/simulated_rirs/mediumroom/rir_list" \
    --rir-set-parameters "0.25, RIRS_NOISES/simulated_rirs/largeroom/rir_list" \
    --rir-set-parameters "0.25, RIRS_NOISES/real_rirs_isotropic_noises/rir_list" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 8000 \
    --include-original-data true \
    data/${clean_data_dir} data/${clean_data_dir}_rvb${num_data_reps}
fi


if [ $stage -le 4 ]; then
  mfccdir=mfcc_rvb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in ${clean_data_dir}_rvb${num_data_reps}; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # do volume-perturbation on the training data prior to extracting hires
    # features; this helps make trained nnets more invariant to test data volume.
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi


# ivector extractor training
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/train_100k_nodup_hires \
    data/lang_nosp exp/tri2_ali_100k_nodup $iv_dir/tri3b
fi

train_set=${clean_data_dir}_rvb${num_data_reps}

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  utils/subset_data_dir.sh data/${train_set}_hires 30000 data/${train_set}_30k_hires
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_30k_hires 512 $iv_dir/tri3b $iv_dir/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  utils/subset_data_dir.sh data/${train_set}_hires 100000 data/${train_set}_100k_hires
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_100k_hires $iv_dir/diag_ubm $iv_dir/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  # handle per-utterance decoding well (iVector starts at zero).

  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_max2_hires $iv_dir/extractor $iv_dir/ivectors_${train_set} || exit 1;
  
  for data_set in eval2000; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires $iv_dir/extractor $iv_dir/ivectors_$data_set || exit 1;
  done
fi

exit 0;

