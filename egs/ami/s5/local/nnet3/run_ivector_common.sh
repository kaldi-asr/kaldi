#!/bin/bash

set -e -o pipefail

# this script contains some common (shared) parts of the run_nnet*.sh scripts.
# speed perturbation is done for the training data

stage=0
mic=ihm
num_threads_ubm=32
nj=10
use_ihm_ali=false
use_sat_alignments=true
min_seg_len=1.55
extractor=
affix=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

train_set=train${affix}

if [ "$use_sat_alignments" == "true" ]; then
  gmm=tri4a${affix}
else
  gmm=tri3a${affix}
fi

if [ "$use_ihm_ali" == "true" ]; then
  if [ "$mic" == "ihm" ]; then
    echo "This is an IHM setup, using the use_ihm_ali=true options does not make sense. Rerun with use_ihm_ali=false" && exit 1;
  fi
  # prepare the parallel data directory ${mic}_clean_ali
  # generate alignments from the perturbed parallel data
  local/nnet3/prepare_parallel_perturbed_alignments.sh --stage $stage \
                                                       --mic $mic \
                                                       --new-mic ${mic}_cleanali \
                                                       --use-sat-alignments $use_sat_alignments \ 
                                                       --affix $affix \
                                                       --min-seg-len $min_seg_len

  # we are going to modify the mic name as changing the alignments
  # changes the ivector extractor
  mic=${mic}_cleanali
  ali_dir=exp/ihm/${gmm}_${mic}_${train_set}_parallel_sp_min${min_seg_len}_ali
else
  # prepare the perturbed data directory and generate alignments
  local/nnet3/prepare_perturbed_alignments.sh --stage $stage --mic $mic \
                                              --use-sat-alignments $use_sat_alignments \
                                              --affix $affix \
                                              --min-seg-len $min_seg_len

  ali_dir=exp/$mic/${gmm}_${mic}_${train_set}_sp_min${min_seg_len}_ali
fi

if [ $stage -le 4 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc_${mic}_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev eval; do
    utils/copy_data_dir.sh data/$mic/$datadir data/$mic/${datadir}_hires
    if [ "$datadir" == "${train_set}_sp" ]; then
      utils/data/perturb_data_dir_volume.sh data/$mic/${datadir}_hires
    fi

    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/$mic/${datadir}_hires exp/make_${mic}_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$mic/${datadir}_hires exp/make_${mic}_hires/$mic/$datadir $mfccdir || exit 1;

    utils/fix_data_dir.sh data/$mic/${datadir}_hires
  done
fi

if [ $stage -le 5 ]; then
  steps/cleanup/combine_short_segments.py \
    --old2new-map data/$mic/${train_set}_sp_min${min_seg_len}/old2new_utt_map \
    --input-data-dir data/$mic/${train_set}_sp_hires \
    --output-data-dir data/$mic/${train_set}_sp_min${min_seg_len}_hires
fi

train_set=${train_set}_sp_min${min_seg_len}

if [ -z "$extractor" ]; then
  extractor=exp/$mic/nnet3${affix}/extractor
  if [ $stage -le 6 ]; then
    # Train a system just for its LDA+MLLT transform.  We use --num-iters 13
    # because after we get the transform (12th iter is the last), any further
    # training is pointless.
    steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --realign-iters "" \
      --splice-opts "--left-context=3 --right-context=3" \
      5000 10000 data/$mic/${train_set}_hires data/lang \
      $ali_dir exp/$mic/nnet3${affix}/tri5
  fi


  if [ $stage -le 6 ]; then
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
      --num-frames 700000 \
      --num-threads $num_threads_ubm \
      data/$mic/${train_set}_hires 512 exp/$mic/nnet3${affix}/tri5 exp/$mic/nnet3${affix}/diag_ubm
  fi

  if [ $stage -le 7 ]; then
    # iVector extractors can in general be sensitive to the amount of data, but
    # this one has a fairly small dim (defaults to 100)
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      data/$mic/${train_set}_hires exp/$mic/nnet3${affix}/diag_ubm exp/$mic/nnet3${affix}/extractor || exit 1;
  fi
fi

if [ $stage -le 8 ]; then
  rm -f exp/$mic/nnet3${affix}/.error 2>/dev/null
  ivectordir=exp/$mic/nnet3${affix}/ivectors_${train_set}_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/$mic/${train_set}_hires data/$mic/${train_set}_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/$mic/${train_set}_hires_max2 \
    $extractor \
    $ivectordir \
    || touch exp/$mic/nnet3${affix}/.error
  [ -f exp/$mic/nnet3${affix}/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

if [ $stage -le 9 ]; then
  rm -f exp/$mic/nnet3${affix}/.error 2>/dev/null
  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/$mic/${data}_hires $extractor exp/$mic/nnet3${affix}/ivectors_${data} || touch exp/$mic/nnet3${affix}/.error &
  done
  wait
  [ -f exp/$mic/nnet3${affix}/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

exit 0;
