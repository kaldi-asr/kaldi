#!/usr/bin/env bash
set -euo pipefail
stage=0
clean_set_aug=train_icsiami
clean_set_sp=train_safet
train_set=train_all
test_set=safe_t_dev1
nnet3_affix=
aug_list="reverb music noise babble clean"
nj=65
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm=tri3
gmm_dir=exp/${gmm}_${train_set}
clean_ali_dir=exp/${gmm}_${train_set}_clean_ali
ali_dir=exp/${gmm}_${train_set}_ali_aug

for f in data/${clean_set_sp}/feats.scp data/${clean_set_aug}/feats.scp \
         ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 0 ]; then
  # Adding simulated RIRs to the original data directory
  echo "$0: Preparing data/${clean_set_aug}_reverb directory"
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ ! -f data/$clean_set_aug/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/$clean_set_aug || exit 1;
  fi

  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/$clean_set_aug data/${clean_set_aug}_reverb
fi

if [ $stage -le 1 ]; then
  aug_list="noise_low noise_high clean"
  local/safet_extract_noises.sh data/train_safet
  steps/data/augment_data_dir.py --utt-prefix "noiselow" --modify-spk-id "true" \
    --bg-snrs "5:6:7:8" --num-bg-noises "1" --bg-noise-dir "data/safe_t_noise_filtered" \
    data/${clean_set_aug} data/${clean_set_aug}_noiselow

  steps/data/augment_data_dir.py --utt-prefix "noisehigh" --modify-spk-id "true" \
    --bg-snrs "0:1:2:3:4" --num-bg-noises "1" --bg-noise-dir "data/safe_t_noise_filtered" \
    data/${clean_set_aug} data/${clean_set_aug}_noisehigh

  utils/combine_data.sh data/${clean_set_aug}_aug data/${clean_set_aug} data/${clean_set_aug}_noiselow data/${clean_set_aug}_noisehigh data/${clean_set_aug}_reverb
fi

if [ $stage -le 2 ]; then
  echo "$0: Extracting low-resolution MFCCs for the augmented data. Useful for generating alignments"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${clean_set_aug}_aug
  steps/compute_cmvn_stats.sh data/${clean_set_aug}_aug
  utils/fix_data_dir.sh data/${clean_set_aug}_aug
fi

if [ $stage -le 3 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh data/${clean_set_sp} data/${clean_set_sp}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data/${clean_set_sp}_sp"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 75 data/${clean_set_sp}_sp
  steps/compute_cmvn_stats.sh data/${clean_set_sp}_sp
  utils/fix_data_dir.sh data/${clean_set_sp}_sp
fi

if [ $stage -le 4 ]; then
  for datadir in ${clean_set_aug}_aug ${clean_set_sp}_sp; do

    echo "$0: creating high-resolution MFCC features for $datadir"
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires

    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 5 ]; then
  echo "$0: obtain the alignment for the clean data"
  utils/data/combine_data.sh data/${train_set}_clean data/${clean_set_sp}_sp data/${clean_set_aug}
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_clean data/lang_nosp_test exp/${gmm}_${train_set} $clean_ali_dir

  echo "$0: aligning augmented data with clean data"
  aug_list="reverb noiselow noisehigh clean"
  utils/data/combine_data.sh data/${train_set}_aug data/${clean_set_sp}_sp data/${clean_set_aug}_aug
  include_original=false
  prefixes=""
  for n in $aug_list; do
    if [ "$n" == "reverb" ]; then
      prefixes="$prefixes "reverb1
    elif [ "$n" != "clean" ]; then
      prefixes="$prefixes "$n
    else
      include_original=true
    fi
  done
  echo "$0: Creating alignments of aug data by copying alignments of clean data"
  steps/copy_ali_dir.sh --nj $nj --cmd "$train_cmd" \
    --include-original "$include_original" --prefixes "$prefixes" \
    data/${train_set}_aug $clean_ali_dir $ali_dir

  utils/data/combine_data.sh data/${train_set}_aug_hires data/${clean_set_sp}_sp_hires data/${clean_set_aug}_aug_hires
fi

if [ $stage -le 6 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm
  num_utts_total=$(wc -l <data/${train_set}_aug_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_aug_hires \
     $num_utts ${temp_data_root}/${train_set}_aug_hires_subset
  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_aug_hires_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_aug_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 7 ]; then
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj \
     data/${train_set}_aug_hires exp/nnet3${nnet3_affix}/diag_ubm \
     exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_aug_hires
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_aug_hires ${temp_data_root}/${train_set}_aug_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_aug_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir
fi
