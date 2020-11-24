#!/usr/bin/env bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
nj=100
train_set=train_all
gmm=tri3
nnet3_affix=_all
extractor=
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

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

#if [ $stage -le 2 ]; then
#  echo "$0: aligning with the perturbed low-resolution data"
#  steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
#    data/${train_set}_sp data/lang_nosp_test $gmm_dir $ali_dir || exit 1
#fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b1{5,6,8,9}/$USER/kaldi-data/mfcc/opensat-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;

  for datadir in ${train_set}_sp; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires data/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
    data/${train_set}_sp_hires_max2 $extractor $ivectordir
fi
exit 0
