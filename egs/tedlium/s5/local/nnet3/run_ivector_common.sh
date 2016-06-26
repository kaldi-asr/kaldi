#!/bin/bash

# This is based on:
# swbd/s5c/local/nnet3/run_ivector_common.sh and
# tedlium/s5/local/online/run_nnet2_ms_perturbed.sh
# see the chain docs for general direction on what training is doing!

set -u -e -o pipefail

stage=1
generate_alignments=true # false if doing ctc training
min_seg_len=1.55
affix=
extractor=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3${affix}
# perturb the data
train_set=train${affix}
if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturb the normal data to get the alignment

  utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
  utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp2
  utils/combine_data.sh data/${train_set}_tmp data/temp1 data/temp2
  utils/validate_data_dir.sh --no-feats data/${train_set}_tmp
  rm -r data/temp1 data/temp2

  mfccdir=mfcc_perturbed
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
    data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit1;
  utils/fix_data_dir.sh data/${train_set}_tmp
  
  utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${train_set} data/temp0
  utils/combine_data.sh data/${train_set}_sp data/${train_set}_tmp data/temp0
  utils/fix_data_dir.sh data/${train_set}_sp
  rm -r data/temp0 data/${train_set}_tmp
fi

train_set_sp=${train_set}_sp

train_set=${train_set_sp}

if [ ! -z "$min_seg_len" ]; then
  if [ $stage -le 2 ]; then
    steps/cleanup/combine_short_segments.py \
      --minimum-duration $min_seg_len \
      --input-data-dir data/${train_set_sp} \
      --output-data-dir data/${train_set_sp}_min$min_seg_len
  fi
  train_set_sp=${train_set_sp}_min${min_seg_len}
fi

if [ $stage -le 3 ] && [ "$generate_alignments" == "true" ]; then
  # obtain the alignment of the pertubed data
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/${train_set_sp} data/lang_nosp exp/tri3${affix} exp/tri3${affix}_ali_${train_set_sp} || exit 1
fi

if [ $stage -le 4 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set; do
    data_dir=data/${dataset}_hires
    utils/copy_data_dir.sh data/$dataset $data_dir
    utils/data/perturb_data_dir_volume.sh $data_dir


    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      $data_dir exp/make_hires/$dataset $mfccdir
    steps/compute_cmvn_stats.sh $data_dir exp/make_hires/$dataset $mfccdir
    utils/fix_data_dir.sh $data_dir # remove segments with problems
  done

  for dataset in dev test; do
    data_dir=data/${dataset}_hires
    utils/copy_data_dir.sh data/$dataset $data_dir

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      $data_dir exp/make_hires/$dataset $mfccdir
    steps/compute_cmvn_stats.sh $data_dir exp/make_hires/$dataset $mfccdir
    utils/fix_data_dir.sh $data_dir # remove segments with problems
  done
fi

if [ ! -z "$min_seg_len" ]; then
  if [ $stage -le 5 ]; then
    steps/cleanup/combine_short_segments.py \
      --old2new-map data/$train_set_sp/old2new_utt_map \
      --input-data-dir data/${train_set}_hires \
      --output-data-dir data/${train_set_sp}_hires
  fi
fi

if [ -z "$extractor" ]; then
  # ivector extractor training
  if [ $stage -le 6 ]; then
    # We need to build a small system just because we need the LDA+MLLT transform
    # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
    # the transform (12th iter is the last), any further training is pointless.
    # this decision is based on fisher_english
    # Note: We do NOT use speed-perturbed data in this step.
    steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --splice-opts "--left-context=3 --right-context=3" \
      5000 10000 data/${train_set_sp}_hires \
      data/lang_nosp exp/tri3${affix}_ali_${train_set_sp} exp/nnet3${affix}/tri3b
  fi

  if [ $stage -le 7 ]; then
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
      data/${train_set_sp}_hires 512 exp/nnet3${affix}/tri3b exp/nnet3${affix}/diag_ubm
  fi

  if [ $stage -le 8 ]; then
      steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
          data/${train_set_sp}_hires exp/nnet3${affix}/diag_ubm exp/nnet3${affix}/extractor || exit 1;
  fi

  extractor=exp/nnet3${affix}/extractor
fi

ivector_dir=`dirname $extractor`

if [ $stage -le 9 ]; then
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set_sp}_hires \
        data/${train_set_sp}_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${train_set_sp}_hires_max2 $extractor $ivector_dir/ivectors_${train_set_sp} || exit 1

    for data_set in dev test; do
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
        data/${data_set}_hires $extractor $ivector_dir/ivectors_${data_set} || exit 1;
    done
fi

