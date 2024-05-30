#!/usr/bin/env bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
nj=60
train_set=train_all
gmm=tri3
nnet3_affix=_all

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}_${train_set}
ali_dir=exp/${gmm}_${train_set}_ali_sp

for f in ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ] ; then
  for dset in train_icsiami train_safet; do
    utils/data/perturb_data_dir_speed_3way.sh data/${dset} data/${dset}_sp
    steps/make_mfcc.sh --nj 75 --cmd "$train_cmd" data/${dset}_sp
    steps/compute_cmvn_stats.sh data/${dset}_sp
    utils/fix_data_dir.sh data/${dset}_sp
  done
  utils/data/combine_data.sh data/${train_set}_sp data/train_safet_sp data/train_icsiami_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
    data/${train_set}_sp data/lang_nosp_test $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: creating high-resolution MFCC features"
  for datadir in train_icsiami_sp train_safet_sp; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
  utils/data/combine_data.sh data/${train_set}_sp_hires data/train_safet_sp_hires data/train_icsiami_sp_hires
fi

if [ $stage -le 4 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm
  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
     $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_sp_hires_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj \
     data/${train_set}_sp_hires exp/nnet3${nnet3_affix}/diag_ubm \
     exp/nnet3${nnet3_affix}/extractor || exit 1;
fi


if [ $stage -le 6 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/opensat-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi


  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
    ${temp_data_root}/${train_set}_sp_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir
fi

exit 0
