#!/bin/bash

# this script contains some common (shared) parts of the run_nnet*.sh scripts.

. cmd.sh


stage=0
mic=ihm
num_threads_ubm=32
speed_perturb=true
use_sat_alignments=true

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/$mic/tri4a
  align_script=steps/align_fmllr.sh
else
  gmm_dir=exp/$mic/tri3a
  align_script=steps/align_si.sh
fi

if [ $stage -le 1 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc_${mic}_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in train dev eval; do
    utils/copy_data_dir.sh data/$mic/$datadir data/$mic/${datadir}_hires
    if [ "$datadir" == "train" ]; then
      dir=data/$mic/train_hires
      cat $dir/wav.scp | python -c "
import sys, os, subprocess, re, random
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $dir/wav.scp_scaled || exit 1;
     mv $dir/wav.scp $dir/wav.scp_nonorm
     mv $dir/wav.scp_scaled $dir/wav.scp
    fi

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/$mic/${datadir}_hires exp/make_${mic}_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$mic/${datadir}_hires exp/make_${mic}_hires/$mic/$datadir $mfccdir || exit 1;
  done

fi

if [ $stage -le 2 ]; then
  # Train a system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/$mic/train_hires data/lang \
    ${gmm_dir}_ali exp/$mic/nnet3/tri5
fi


if [ $stage -le 3 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    data/$mic/train_hires 512 exp/$mic/nnet3/tri5 exp/$mic/nnet3/diag_ubm
fi

if [ $stage -le 4 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100)
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/$mic/train_hires exp/$mic/nnet3/diag_ubm exp/$mic/nnet3/extractor || exit 1;
fi

if [ $stage -le 5 ] && [ "$speed_perturb" == "true" ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 data/$mic/train data/$mic/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/$mic/train data/$mic/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/$mic/train data/$mic/temp3
  utils/combine_data.sh --extra-files utt2uniq data/$mic/train_sp data/$mic/temp1 data/$mic/temp2 data/$mic/temp3
  rm -r data/$mic/temp1 data/$mic/temp2 data/$mic/temp3

  mfccdir=mfcc_${mic}_perturbed
  for x in train_sp; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
      data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/$mic/train_sp

  $align_script --nj $nj --cmd "$train_cmd" \
    data/$mic/train_sp data/lang $gmm_dir ${gmm_dir}_sp_ali || exit 1

  #Now perturb the high resolution daa
  utils/copy_data_dir.sh data/$mic/train_sp data/$mic/train_sp_hires
  mfccdir=mfcc_${mic}_perturbed_hires
  for x in train_sp_hires; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf \
      data/$mic/$x exp/make_${mic}_hires/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$mic/$x exp/make_${mic}_hires/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/$mic/train_sp_hires
fi

if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
else
  train_set=train
fi

if [ $stage -le 6 ]; then
  rm exp/$mic/nnet3/.error 2>/dev/null
  ivectordir=exp/$mic/nnet3/ivectors_${train_set}_hires
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
    exp/$mic/nnet3/extractor \
    exp/$mic/nnet3/ivectors_${train_set}_hires \
    || touch exp/$mic/nnet3/.error
  [ -f exp/$mic/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

if [ $stage -le 7 ]; then
  rm -f exp/$mic/nnet3/.error 2>/dev/null
  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/$mic/${data}_hires exp/$mic/nnet3/extractor exp/$mic/nnet3/ivectors_${data} || touch exp/$mic/nnet3/.error &
  done
  wait
  [ -f exp/$mic/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

exit 0;
