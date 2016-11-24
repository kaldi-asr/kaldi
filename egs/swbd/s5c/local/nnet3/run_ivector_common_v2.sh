#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
speaker_perturb=true
lpc_order=100
filter_nj=30
spkf_per_spk=3
perturb_suffix=""

. ./path.sh
. ./utils/parse_options.sh

mkdir -p nnet3
# perturbed data preparation
train_set=train_nodup

if $speed_perturb; then
  perturb_suffix="_sp"
fi

if $speaker_perturb; then
  perturb_suffix=$perturb_suffix"_fp"
fi

if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    echo "speed perturb the data"
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in train_nodup; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp_sp1
      utils/perturb_data_dir_speed.sh 0.95 data/${datadir} data/temp_sp2
      utils/perturb_data_dir_speed.sh 1.05 data/${datadir} data/temp_sp3
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp_sp4

      utils/combine_data.sh data/${datadir}_temp_sp data/temp_sp1 data/temp_sp2 data/temp_sp3 data/temp_sp4
      utils/validate_data_dir.sh --no-feats data/${datadir}_temp_sp
      rm -r data/temp_sp1 data/temp_sp2 data/temp_sp3 data/temp_sp4

      if [ "$speaker_perturb" == "true" ]; then
        echo "speaker perturbation of data"
        utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp_sp0
        utils/combine_data.sh data/${datadir}_sp data/${datadir}_temp_sp data/temp_sp0
        utils/fix_data_dir.sh data/${datadir}_sp
        
        # compute filter correspond to different speed perturbed speaker.
        spk_filters=spkfilters
        mkdir -p $spk_filters
        utils/split_data.sh data/${datadir}_sp $filter_nj
        echo $filter_nj > data/${datadir}_sp/num_filter_jobs

        $decode_cmd JOB=1:$filter_nj data/${datadir}_sp/split$filter_nj/compute_filter.JOB.log \
          compute-filter --lpc-order=$lpc_order scp:data/${datadir}_sp/split$filter_nj/JOB/wav.scp \
            ark,scp:$spk_filters/spk_filter.JOB.ark,$spk_filters/spk_filter.JOB.scp || exit 1;
        
        # combine filters.scp files together 
        for n in $(seq $filter_nj); do
          cat $spk_filters/spk_filter.$n.scp || exit 1;
        done > data/${datadir}_sp/spk_filter.scp
        echo "Finished generating filters per speakers."

        echo "Perturb data using speaker perturbation."
        utils/perturb_data_signal_v2.sh $spkf_per_spk 'fp' data/${datadir}_sp data/${datadir}_temp_sp_fp
        utils/validate_data_dir.sh --no-feats data/${datadir}_temp_sp_fp
      fi

      echo "perturb_suffix=$perturb_suffix "
      mfccdir=mfcc_perturbed
      echo "Generating features using perturbed data"
      steps/make_mfcc.sh --cmd "$decode_cmd" --nj 50 \
        data/${datadir}_temp${perturb_suffix} exp/make_mfcc/${datadir}_temp${perturb_suffix} $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_temp${perturb_suffix} exp/make_mfcc/${datadir}_temp${perturb_suffix} $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_temp${perturb_suffix}

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}${perturb_suffix} data/${datadir}_temp${perturb_suffix} data/temp0
      utils/fix_data_dir.sh data/${datadir}${perturb_suffix}
      rm -r data/temp0 data/${datadir}_temp${perturb_suffix}
    done
  fi

  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/train_nodup${perturb_suffix} data/lang_nosp exp/tri4 exp/tri4_ali_nodup${perturb_suffix} || exit 1
  fi
fi

train_set=train_nodup${perturb_suffix}
if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri2_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set train_100k_nodup; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # scale the waveforms, this is useful as we don't use CMVN
    data_dir=data/${dataset}_hires
    cat $data_dir/wav.scp | python -c "
import sys, os, subprocess, re, random
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $data_dir/wav.scp_scaled || exit 1;
    mv $data_dir/wav.scp_scaled $data_dir/wav.scp

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
  if false; then #300
  for dataset in eval2000 train_dev; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
  fi #300
  # Take the first 30k utterances (about 1/8th of the data) this will be used
  # for the diagubm training
  utils/subset_data_dir.sh --first data/${train_set}_hires 30000 data/${train_set}_30k_hires
  local/remove_dup_utts.sh 200 data/${train_set}_30k_hires data/${train_set}_30k_nodup_hires  # 33hr
fi
if false; then #400
# ivector extractor training
if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/train_100k_nodup_hires \
    data/lang_nosp exp/tri2_ali_100k_nodup exp/nnet3/tri3b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_30k_nodup_hires 512 exp/nnet3/tri3b exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_100k_nodup_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi
fi #400

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_max2_hires exp/nnet3/extractor exp/nnet3/ivectors_$train_set || exit 1;

  for data_set in eval2000 train_dev; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires exp/nnet3/extractor exp/nnet3/ivectors_$data_set || exit 1;
  done
fi

exit 0;
