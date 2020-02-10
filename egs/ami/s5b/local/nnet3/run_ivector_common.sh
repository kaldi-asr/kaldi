#!/usr/bin/env bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

stage=0
mic=ihm
nj=30
min_seg_len=1.55  # min length in seconds... we do this because chain training
                  # will discard segments shorter than 1.5 seconds.  Must remain in sync with
                  # the same option given to prepare_lores_feats.sh.
train_set=train   # you might set this to e.g. train_cleaned.
gmm=tri3          # This specifies a GMM-dir from the features of the type you're training the system on;
                  # it should contain alignments for 'train_set'.

num_threads_ubm=32
ivector_transform_type=lda
nnet3_affix=_cleaned     # affix for exp/$mic/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/$mic/nnet3_cleaned or whatever.
hires_suffix=
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


gmmdir=exp/${mic}/${gmm}


for f in data/${mic}/${train_set}/feats.scp ; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done



if [ $stage -le 2 ] && [ -f data/$mic/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: data/$mic/${train_set}_sp_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "$0: preparing directory for speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh data/${mic}/${train_set} data/${mic}/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/$mic/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev eval; do
    utils/copy_data_dir.sh data/$mic/$datadir data/$mic/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/$mic/${train_set}_sp_hires

  for datadir in ${train_set}_sp dev eval; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires$hires_suffix.conf \
      --cmd "$train_cmd" data/$mic/${datadir}_hires
    steps/compute_cmvn_stats.sh data/$mic/${datadir}_hires
    utils/fix_data_dir.sh data/$mic/${datadir}_hires
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  utils/data/combine_short_segments.sh \
     data/${mic}/${train_set}_sp_hires $min_seg_len data/${mic}/${train_set}_sp_hires_comb

  # just copy over the CMVN to avoid having to recompute it.
  cp data/${mic}/${train_set}_sp_hires/cmvn.scp data/${mic}/${train_set}_sp_hires_comb/
  utils/fix_data_dir.sh data/${mic}/${train_set}_sp_hires_comb/
fi

if [ $stage -le 4 ]; then
  echo "$0: selecting segments of hires training data that were also present in the"
  echo " ... original training data."

  # note, these data-dirs are temporary; we put them in a sub-directory
  # of the place where we'll make the alignments.
  temp_data_root=exp/$mic/nnet3${nnet3_affix}/tri5
  mkdir -p $temp_data_root

  utils/data/subset_data_dir.sh --utt-list data/${mic}/${train_set}/feats.scp \
          data/${mic}/${train_set}_sp_hires $temp_data_root/${train_set}_hires

  # note: essentially all the original segments should be in the hires data.
  n1=$(wc -l <data/${mic}/${train_set}/feats.scp)
  n2=$(wc -l <$temp_data_root/${train_set}_hires/feats.scp)
  if [ $n1 != $n1 ]; then
    echo "$0: warning: number of feats $n1 != $n2, if these are very different it could be bad."
  fi

  case $ivector_transform_type in
    lda)
      if [ ! -f ${gmmdir}/final.mdl ]; then
        echo "$0: expected file ${gmmdir}/final.mdl to exist"
        exit 1;
      fi
      echo "$0: training a system on the hires data for its LDA+MLLT transform, in order to produce the diagonal GMM."
      if [ -e exp/$mic/nnet3${nnet3_affix}/tri5/final.mdl ]; then
        # we don't want to overwrite old stuff, ask the user to delete it.
        echo "$0: exp/$mic/nnet3${nnet3_affix}/tri5/final.mdl already exists: "
        echo " ... please delete and then rerun, or use a later --stage option."
        exit 1;
      fi
      steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 7 --mllt-iters "2 4 6" \
        --splice-opts "--left-context=3 --right-context=3" \
        3000 10000 $temp_data_root/${train_set}_hires data/lang \
        $gmmdir exp/$mic/nnet3${nnet3_affix}/tri5
      ;;
    pca)
      echo "$0: computing a PCA transform from the hires data."
      steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
        --splice-opts "--left-context=3 --right-context=3" \
        --max-utts 10000 --subsample 2 \
        $temp_data_root/${train_set}_hires \
        exp/$mic/nnet3${nnet3_affix}/tri5
      ;;
    *) echo "$0: invalid iVector transform type $ivector_transform_type" && exit 1;
  esac
fi

if [ $stage -le 5 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p exp/$mic/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/$mic/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  # we don't use the _comb data for this as there is no need for compatibility with
  # the alignments, and using the non-combined data is more efficient for I/O
  # (no messing about with piped commands).
  num_utts_total=$(wc -l <data/$mic/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/$mic/${train_set}_sp_hires \
      $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    exp/$mic/nnet3${nnet3_affix}/tri5 exp/$mic/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/$mic/${train_set}_sp_hires exp/$mic/nnet3${nnet3_affix}/diag_ubm exp/$mic/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${mic}/${train_set}_sp_hires_comb ${temp_data_root}/${train_set}_sp_hires_comb_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_comb_max2 \
    exp/$mic/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp) or small-segment concatenation (comb).
  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
      data/${mic}/${data}_hires exp/$mic/nnet3${nnet3_affix}/extractor \
      exp/$mic/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi

exit 0;
