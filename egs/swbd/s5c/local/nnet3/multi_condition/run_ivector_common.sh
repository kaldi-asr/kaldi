#!/bin/bash 
#set -e
# this script is based on local/nnet3/run_ivector_common.sh
# but it operates on corrupted training/dev/test data sets

. cmd.sh

stage=1
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_data_reps=1
clean_data_dir=train_nodup_sp
iv_dir=exp/nnet3_rvb

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p $iv_dir
train_set=${clean_data_dir}_rvb${num_data_reps}

if [ $stage -le 1 ]; then
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # corrupt the data to generate reverberated data 
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
    data/${clean_data_dir} data/${train_set}
fi


if [ $stage -le 2 ]; then
  mfccdir=mfcc_rvb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set; do
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
  # Here we want to build a 200k system, half from the reverberated set and half from the original set
  local/nnet3/multi_condition/copy_ali_dir.sh --utt-prefix "rev1_sp1.0-" exp/tri2_ali_100k_nodup exp/tri2_ali_100k_nodup_rvb || exit 1;
  local/nnet3/multi_condition/copy_ali_dir.sh --utt-prefix "rev0_sp1.0-" exp/tri2_ali_100k_nodup exp/tri2_ali_100k_nodup_clean || exit 1;

  # want the 100k subset to exactly match train_100k, since we'll use its alignments.
  awk -v p='rev1_sp1.0-' '{printf "%s%s\n", p, $1}' data/train_100k_nodup/utt2spk > uttlist
  utils/subset_data_dir.sh --utt-list uttlist \
    data/${train_set}_hires data/${train_set}_100k_hires
  rm uttlist

  # Mix the 100k original data and the 100k reverberated data
  utils/copy_data_dir.sh --spk-prefix "rev0_sp1.0-" --utt-prefix "rev0_sp1.0-" data/train_100k_nodup_hires data/train_100k_nodup_hires_tmp
  utils/combine_data.sh data/${train_set}_200k_mix_hires data/train_100k_nodup_hires_tmp data/${train_set}_100k_hires
  rm -r data/train_100k_nodup_hires_tmp

  # combine the alignment for mixed data
  steps/combine_ali_dirs.sh --num-jobs 30 data/${train_set}_200k_mix_hires exp/tri2_ali_200k_mix exp/tri2_ali_100k_nodup_clean exp/tri2_ali_100k_nodup_rvb || exit 1;
  rm -r exp/tri2_ali_100k_nodup_clean exp/tri2_ali_100k_nodup_rvb

  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/${train_set}_200k_mix_hires \
    data/lang_nosp exp/tri2_ali_200k_mix $iv_dir/tri3b
fi

if [ $stage -le 6 ]; then
  utils/copy_data_dir.sh --spk-prefix "rev0_" --utt-prefix "rev0_" data/${clean_data_dir}_30k_nodup_hires data/${clean_data_dir}_30k_nodup_hires_tmp
  # want the reverberated 30k subset to exactly match clean 30k, since we'll use its alignments.
  awk -v p='rev1_' '{printf "%s%s\n", p, $1}' data/${clean_data_dir}_30k_nodup_hires/utt2spk > uttlist
  utils/subset_data_dir.sh --utt-list uttlist \
    data/${train_set}_hires data/${train_set}_30k_hires
  rm uttlist

  # Mix the 30k original data and the 30k reverberated data
  utils/combine_data.sh data/${train_set}_60k_mix_hires data/${clean_data_dir}_30k_nodup_hires_tmp data/${train_set}_30k_hires
  rm -r data/${clean_data_dir}_30k_nodup_hires_tmp

  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_60k_mix_hires 512 $iv_dir/tri3b $iv_dir/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_200k_mix_hires $iv_dir/diag_ubm $iv_dir/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  # handle per-utterance decoding well (iVector starts at zero).

  # Mix all the original data and all the reverberated data
  utils/copy_data_dir.sh --spk-prefix "rev0_" --utt-prefix "rev0_" data/${clean_data_dir}_hires data/${clean_data_dir}_hires_clean
  utils/combine_data.sh data/${train_set}_mix_hires data/${clean_data_dir}_hires_clean data/${train_set}_hires

  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_mix_hires data/${train_set}_mix_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_mix_max2_hires $iv_dir/extractor $iv_dir/ivectors_${train_set}_mix || exit 1;
  
  for data_set in eval2000; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires $iv_dir/extractor $iv_dir/ivectors_$data_set || exit 1;
  done
fi

exit 0;

