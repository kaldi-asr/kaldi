#!/usr/bin/env bash
set -euo pipefail
stage=0
train_set=train_icsiami
test_sets=safe_t_dev1
gmm=tri3
nnet3_affix=
aug_list="reverb music noise babble clean"
use_ivectors=true
num_reverb_copies=1
clean_ali=tri3b_ali_train_clean_5

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}_${train_set}
ali_dir=exp/${gmm}_${train_set}_ali

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

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

  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_reverb_copies \
    --source-sampling-rate 16000 \
    data/$train_set data/${train_set}_reverb
fi

if [ $stage -le 1 ]; then
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # We will use them as additive noises for data augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 --use-vocals "true" \
        /export/corpora/JHU/musan data
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
    --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
    data/${train_set} data/${train_set}_noise
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id "true" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
    data/${train_set} data/${train_set}_music
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id "true" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/musan_speech" \
    data/${train_set} data/${train_set}_babble

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
  echo "$0: Extracting low-resolution MFCCs for the augmented data. Useful for generating alignments"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 75 data/${train_set}_aug
  steps/compute_cmvn_stats.sh data/${train_set}_aug
  utils/fix_data_dir.sh data/${train_set}_aug
fi

if [ $stage -le 3 ]; then
  echo "$0: creating high-resolution MFCC features"
  for datadir in ${train_set}_aug ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires
  done

  for datadir in ${train_set}_aug ${test_sets}; do
    steps/make_mfcc.sh --nj 75 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 4 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 5 ]; then
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data
  for datadir in ${train_set}_sp; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires
  for datadir in ${train_set}_sp; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

if [ $stage -le 6 ]; then
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
      include_original=true
    fi
  done
  echo "$0: Creating alignments of aug data by copying alignments of clean data"
  steps/copy_ali_dir.sh --nj 10 --cmd "$train_cmd" \
    --include-original "$include_original" --prefixes "$prefixes" \
    data/${train_set}_aug exp/${clean_ali} exp/${clean_ali}_aug
fi

if [ $stage -le 6 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp_test $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 5 ]; then
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
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 75 \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_aug_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 75 \
     data/${train_set}_aug_hires exp/nnet3${nnet3_affix}/diag_ubm \
     exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_aug_hires
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_aug_hires ${temp_data_root}/${train_set}_aug_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 75 \
    ${temp_data_root}/${train_set}_aug_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir
  for data in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 75 \
      data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi
