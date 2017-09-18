#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
snrs="20:15:10"
num_data_reps=3
ali_dir=exp/
db_string="'air' 'rwcp' 'rvb2014'" # RIR dbs to be used in the experiment
                                      # only dbs used for ASpIRE submission system have been used here
RIR_home=db/RIR_databases/ # parent directory of the RIR databases files
download_rirs=true # download the RIR databases from the urls or assume they are present in the RIR_home directory



[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

# perturbed data preparation
train_set=train
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed
    for datadir in train; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
      utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
      rm -r data/temp1 data/temp2

      featdir=plp_perturbed
      if $use_pitch; then
        steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/${datadir}_tmp exp/make_plp_pitch/${datadir}_tmp $featdir
      else
        steps/make_plp.sh --cmd "$train_cmd" --nj $train_nj data/${datadir}_tmp exp/make_plp/${datadir}_tmp $featdir
      fi

      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_plp/${datadir}_tmp $featdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      utils/fix_data_dir.sh data/${datadir}_sp
      rm -r data/temp0 data/${datadir}_tmp
    done
  fi

  train_set=train_sp
  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh \
      --nj 70 --cmd "$train_cmd" \
      --boost-silence $boost_sil \
      data/$train_set data/langp/tri5_ali exp/tri5 exp/tri5_ali_sp || exit 1
    touch exp/tri5_ali_sp/.done
  fi
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/egs/kaldi-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri2_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set ; do
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

fi

# check if the required tools are present
$KALDI_ROOT/egs/aspire/s5/local/multi_condition/check_version.sh || exit 1;
mkdir -p exp/nnet3_multicondition
if [ $stage -le 4 ]; then
  # prepare the impulse responses
  local/multi_condition/prepare_impulses_noises.sh --log-dir exp/make_reverb/log \
    --db-string "$db_string" \
    --download-rirs $download_rirs \
    --RIR-home $RIR_home \
    data/impulses_noises || exit 1;
fi

if [ $stage -le 5 ]; then
  # corrupt the training data to generate multi-condition data
  for data_dir in train_sp; do
    num_reps=$num_data_reps
    reverb_data_dirs=
    for i in `seq 1 $num_reps`; do
      cur_dest_dir=" data/temp_${data_dir}_${i}"
      $KALDI_ROOT/egs/aspire/s5/local/multi_condition/reverberate_data_dir.sh --random-seed $i \
        --snrs "$snrs" --log-dir exp/make_corrupted_wav \
        data/${data_dir}  data/impulses_noises $cur_dest_dir
      reverb_data_dirs+=" $cur_dest_dir"
    done
    utils/combine_data.sh --extra-files utt2uniq data/${data_dir}_mc data/${data_dir} $reverb_data_dirs
    rm -rf $reverb_data_dirs
  done
fi

if [ $stage -le 6 ]; then
  # copy the alignments for the newly created utterance ids
  ali_dirs=
  for i in `seq 1 $num_data_reps`; do
    local/multi_condition/copy_ali_dir.sh --utt-prefix "rev${i}_" exp/tri5_ali_sp exp/tri5_ali_sp_temp_$i || exit 1;
    ali_dirs+=" exp/tri5_ali_sp_temp_$i"
  done
    local/multi_condition/copy_ali_dir.sh exp/tri5_ali_sp exp/tri5_ali_sp_copy || exit 1;
  ali_dirs+=" exp/tri5_ali_sp_copy"
  utils/combine_ali_dirs.sh --num-jobs 32 \
    data/train_sp_mc exp/tri5_ali_sp_mc $ali_dirs || exit 1;
  rm -rf $ali_dirs
fi

train_set=train_sp_mc
if [ $stage -le 7 ]; then
  mfccdir=mfcc_reverb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel_reverb-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$mfccdir/storage $mfccdir/storage
  fi
  for data_dir in $train_set; do
    utils/copy_data_dir.sh data/$data_dir data/${data_dir}_hires
    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${data_dir}_hires \
        exp/make_reverb_hires/${data_dir} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${data_dir}_hires exp/make_reverb_hires/${data_dir} $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${data_dir}_hires
    utils/validate_data_dir.sh data/${data_dir}_hires
  done
fi

# ivector extractor training
if [ $stage -le 8 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    --boost-silence $boost_sil \
    $numLeavesMLLT $numGaussMLLT data/${train_set}_hires \
    data/langp/tri5_ali exp/tri5_ali_sp_mc exp/nnet3_multicondition/tri3b
fi

if [ $stage -le 9 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_hires 512 exp/nnet3_multicondition/tri3b exp/nnet3_multicondition/diag_ubm
fi

if [ $stage -le 10 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_hires exp/nnet3_multicondition/diag_ubm exp/nnet3_multicondition/extractor || exit 1;
fi

if [ $stage -le 11 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_max2_hires exp/nnet3_multicondition/extractor exp/nnet3_multicondition/ivectors_$train_set || exit 1;

fi

exit 0;
