#!/usr/bin/env bash

set -e -o pipefail

# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
nj=100
train_set=train_mer80  # you might set this to e.g. train.
num_threads_ubm=32
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
extractor=

. cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

for f in data/${train_set}/utt2spk; do 
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 2 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: data/${train_set}_sp_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: preparing directory for speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b{05,06,10,11}/$USER/kaldi-data/egs/mgb2_arabic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for datadir in ${train_set}_sp dev; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ -z "$extractor" ]; then
  if [ $stage -le 4 ]; then
    echo "$0: selecting segments of hires training data that were also present in the"
    echo " ... original training data."

    # note, these data-dirs are temporary; we put them in a sub-directory
    # of the place where we'll make the alignments.
    temp_data_root=exp/nnet3${nnet3_affix}/tri5
    mkdir -p $temp_data_root

    utils/data/subset_data_dir.sh --utt-list data/${train_set}/feats.scp \
            data/${train_set}_sp_hires $temp_data_root/${train_set}_hires

    # note: essentially all the original segments should be in the hires data.
    n1=$(wc -l <data/${train_set}/feats.scp)
    n2=$(wc -l <$temp_data_root/${train_set}_hires/feats.scp)
    if [ $n1 != $n2 ]; then
      echo "$0: warning: number of feats $n1 != $n2, if these are very different it could be bad."
    fi

    echo "$0: training a system on the hires data for its PCA transform, in order to produce the diagonal GMM."
    if [ -e exp/nnet3${nnet3_affix}/pca_transform/final.mat ]; then
      # we don't want to overwrite old stuff, ask the user to delete it.
      echo "$0: exp/nnet3${nnet3_affix}/pca_transform/final.mat already exists: "
      echo " ... please delete and then rerun, or use a later --stage option."
      exit 1;
    fi
    steps/online/nnet2/get_pca_transform.sh \
      --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
       $temp_data_root/${train_set}_hires \
        exp/nnet3${nnet3_affix}/pca_transform
  fi


  if [ $stage -le 5 ]; then
    echo "$0: computing a subset of data to train the diagonal UBM."

    mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
    temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

    # train a diagonal UBM using a subset of about a quarter of the data
    num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
    num_utts=$[$num_utts_total/4]
    utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
        $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

    echo "$0: training the diagonal UBM."
    # Use 512 Gaussians in the UBM.
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
      --num-frames 700000 \
      --num-threads $num_threads_ubm \
      ${temp_data_root}/${train_set}_sp_hires_subset 512 \
      exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
  fi

  if [ $stage -le 6 ]; then
    # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
    # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
    # 100.
    echo "$0: training the iVector extractor"
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      data/${train_set}_sp_hires exp/nnet3${nnet3_affix}/diag_ubm exp/nnet3${nnet3_affix}/extractor || exit 1;
  fi

  extractor=exp/nnet3${nnet3_affix}/extractor
fi

if [ $stage -le 7 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b{05,06,10,11}/$USER/kaldi-data/egs/mgb2_arabic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # With --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_max2 \
    $extractor $ivectordir
  
  for suffix in overlap non_overlap; do
    # Create hires version of the dev dirs
    utils/subset_data_dir.sh --utt-list data/dev_${suffix}/utt2spk \
      data/dev_hires data/dev_${suffix}_hires
    cp data/dev_${suffix}/stm data/dev_${suffix}_hires
  done

  # Also extract iVectors for the test data
  for data in dev dev_non_overlap; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
      data/${data}_hires $extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 9 ]; then
  echo "$0: $feats already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 9 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

exit 0;
