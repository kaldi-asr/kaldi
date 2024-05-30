#!/usr/bin/env bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2014  Tom Ko
# Apache 2.0

# This example script demonstrates how speed perturbation of the data helps the nnet training in the SWB setup.

. ./cmd.sh
set -e
stage=0
train_stage=-10
use_gpu=true
nnet2_online=nnet2_online_ms_p
splice_indexes="layer0/-4:-3:-2:-1:0:1:2:3:4 layer2/-5:-3:3"
common_egs_dir=

. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

dir=exp/$nnet2_online/nnet_a
mkdir -p exp/$nnet2_online

if [ $stage -le 0 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  utils/copy_data_dir.sh data/train data/train_hires
  steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_hires exp/make_hires/train $mfccdir;
  steps/compute_cmvn_stats.sh data/train_hires exp/make_hires/train $mfccdir;

  # Remove the small number of utterances that couldn't be extracted for some
  # reason (e.g. too short; no such file).
  utils/fix_data_dir.sh data/train_hires;

  # Create MFCCs for the eval set
  utils/copy_data_dir.sh data/eval2000 data/eval2000_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
      data/eval2000_hires exp/make_hires/eval2000 $mfccdir;
  steps/compute_cmvn_stats.sh data/eval2000_hires exp/make_hires/eval2000 $mfccdir;
    utils/fix_data_dir.sh data/eval2000_hires  # remove segments with problems

  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  # the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
  # LM training data.   However, they will be in the lexicon, plus speakers
  # may overlap, so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first data/train_hires 4000 data/train_hires_dev ;# 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train_hires $n data/train_hires_nodev ;

  # Take the first 30k utterances (about 1/8th of the data) this will be used
  # for the diagubm training
  utils/subset_data_dir.sh --first data/train_hires_nodev 30000 data/train_hires_30k
  local/remove_dup_utts.sh 200 data/train_hires_30k data/train_hires_30k_nodup  # 33hr

  # create a 100k subset for the lda+mllt training
  utils/subset_data_dir.sh --first data/train_hires_nodev 100000 data/train_hires_100k;
  local/remove_dup_utts.sh 200 data/train_hires_100k data/train_hires_100k_nodup;

  local/remove_dup_utts.sh 300 data/train_hires_nodev data/train_hires_nodup  # 286hr
fi

if [ $stage -le 1 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/train_hires_100k_nodup data/lang exp/tri2_ali_100k_nodup exp/$nnet2_online/tri3b
fi

if [ $stage -le 2 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/train_hires_30k_nodup 512 exp/$nnet2_online/tri3b exp/$nnet2_online/diag_ubm
fi

if [ $stage -le 3 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_hires_100k_nodup exp/$nnet2_online/diag_ubm exp/$nnet2_online/extractor || exit 1;
fi


if [ $stage -le 4 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  utils/perturb_data_dir_speed.sh 0.9 data/train_nodup data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train_nodup data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train_nodup data/temp3
  utils/combine_data.sh data/train_nodup_perturbed data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_perturbed
  for x in train_nodup_perturbed; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_nodup_perturbed
fi

if [ $stage -le 5 ]; then
  #obtain the alignment of the perturbed data
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train_nodup_perturbed data/lang exp/tri4b exp/tri4b_ali_nodup_perturbed || exit 1
fi


if [ $stage -le 6 ]; then
  #Now perturb the high resolution daa
  utils/perturb_data_dir_speed.sh 0.9 data/train_hires_nodup data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train_hires_nodup data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train_hires_nodup data/temp3
  utils/combine_data.sh data/train_hires_nodup_perturbed data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_hires_perturbed
  for x in train_hires_nodup_perturbed; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 70 --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_hires/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_hires/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_hires_nodup_perturbed
fi

if [ $stage -le 7 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires_nodup_perturbed data/train_hires_nodup_perturbed_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_hires_nodup_perturbed_max2 exp/$nnet2_online/extractor exp/$nnet2_online/ivectors_train_hires_nodup_perturbed2 || exit 1;
fi


if [ $stage -le 8 ]; then
  # Because we have a lot of data here and we don't want the training to take
  # too long so we reduce the number of epochs from the defaults (15 + 5) to (5
  # + 2), and the (initial,final) learning rate from the defaults (0.04, 0.004)
  # to (0.01, 0.001).
  # decided to let others run their jobs too (we only have 10 GPUs on our queue
  # at JHU).  The number of parameters is smaller than the baseline system we had in
  # mind (../nnet2/run_5d_gpu.sh, which had pnorm input/output dim 3000/300 and
  # 5 hidden layers, versus our 3000/300 and 5 hidden layers, even though we're
  # training on more data than the baseline system.  The motivation here is that we
  # want to demonstrate the capability of doing real-time decoding, and if the
  # network was too bug we wouldn't be able to decode in real-time using a CPU.
  steps/nnet2/train_pnorm_multisplice2.sh --stage $train_stage \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/$nnet2_online/ivectors_train_hires_nodup_perturbed2 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 6 \
    --num-epochs 2 \
    --add-layers-period 1 \
    --num-hidden-layers 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 3000 \
    --pnorm-output-dim 300 \
    data/train_hires_nodup_perturbed data/lang exp/tri4b_ali_nodup_perturbed $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  for data in eval2000_hires train_hires_dev; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data} exp/$nnet2_online/extractor exp/$nnet2_online/ivectors_${data} || exit 1;
  done
fi


if [ $stage -le 10 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding (the one with --per-utt true)
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    # use already-built graphs.
    for data in eval2000_hires train_hires_dev; do
      steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
          --online-ivector-dir exp/$nnet2_online/ivectors_${data} \
         $graph_dir data/${data} $dir/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi

if [ $stage -le 11 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
      data/lang exp/$nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 12 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000_hires train_hires_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
        "$graph_dir" data/${data} ${dir}_online/decode_${data}_sw1_${lm_suffix} || exit 1;
    done
  done
fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for lm_suffix in tg fsh_tgpr; do
    graph_dir=exp/tri4b/graph_sw1_${lm_suffix}
    for data in eval2000_hires train_hires_dev; do
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
        --per-utt true \
        "$graph_dir" data/${data} ${dir}_online/decode_${data}_sw1_${lm_suffix}_per_utt || exit 1;
    done
  done
fi

exit 0;

# get results on eval2000 with this command:
for x in exp/$nnet2_online/nnet_a/decode_eval2000_*; do grep Sum $x/score_*/*sys  | utils/best_wer.sh; done

# First, this is the baseline.
# This is obtained from running local/online/run_nnet2.sh which calls steps/nnet2/train_pnorm_simple2.sh
# decode on SWBD only
%WER 15.7 | 1831 21395 | 86.1 9.4 4.6 1.8 15.7 54.3 | exp/nnet2_online/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.swbd.filt.sys
%WER 16.1 | 1831 21395 | 85.9 9.8 4.3 2.1 16.1 54.7 | exp/nnet2_online/nnet_a/decode_eval2000_hires_sw1_tg/score_10/eval2000_hires.ctm.swbd.filt.sys
%WER 15.6 | 1831 21395 | 86.0 9.2 4.7 1.7 15.6 53.5 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 16.3 | 1831 21395 | 85.5 9.9 4.6 1.8 16.3 54.0 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_11/eval2000_hires.ctm.swbd.filt.sys
%WER 16.1 | 1831 21395 | 85.9 9.8 4.3 2.1 16.1 54.7 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_tg/score_10/eval2000_hires.ctm.swbd.filt.sys
%WER 16.7 | 1831 21395 | 85.4 10.2 4.3 2.1 16.7 55.3 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_10/eval2000_hires.ctm.swbd.filt.sys

# decode on SWBD+CALLHM
%WER 22.2 | 4459 42989 | 80.2 13.6 6.3 2.4 22.2 60.0 | exp/nnet2_online/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.filt.sys
%WER 23.0 | 4459 42989 | 79.6 14.1 6.3 2.6 23.0 60.8 | exp/nnet2_online/nnet_a/decode_eval2000_hires_sw1_tg/score_11/eval2000_hires.ctm.filt.sys
%WER 22.2 | 4459 42989 | 80.2 13.6 6.3 2.4 22.2 59.9 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.filt.sys
%WER 23.3 | 4459 42989 | 79.2 14.5 6.3 2.6 23.3 60.7 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_11/eval2000_hires.ctm.filt.sys
%WER 23.0 | 4459 42989 | 79.6 14.1 6.2 2.6 23.0 60.8 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_tg/score_11/eval2000_hires.ctm.filt.sys
%WER 24.0 | 4459 42989 | 78.9 15.2 5.9 3.0 24.0 62.0 | exp/nnet2_online/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_10/eval2000_hires.ctm.filt.sys

# Then the following results are obtained from running local/online/run_nnet2_multisplice.sh which calls steps/nnet2/train_pnorm_multisplice2.sh
# decode on SWBD only
%WER 15.0 | 1831 21395 | 86.6 9.0 4.5 1.6 15.0 53.1 | exp/nnet2_online_ms/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.4 | 1831 21395 | 86.3 9.3 4.4 1.7 15.4 54.1 | exp/nnet2_online_ms/nnet_a/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.0 | 1831 21395 | 86.6 8.9 4.5 1.6 15.0 53.0 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.6 | 1831 21395 | 86.1 9.4 4.6 1.7 15.6 53.2 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.5 | 1831 21395 | 86.2 9.4 4.4 1.7 15.5 54.0 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 16.0 | 1831 21395 | 85.8 9.7 4.5 1.8 16.0 53.6 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_12/eval2000_hires.ctm.swbd.filt.sys

# decode on SWBD+CALLHM
%WER 21.8 | 4459 42989 | 80.5 13.3 6.2 2.3 21.8 59.4 | exp/nnet2_online_ms/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_12/eval2000_hires.ctm.filt.sys
%WER 22.5 | 4459 42989 | 79.9 13.9 6.2 2.4 22.5 60.4 | exp/nnet2_online_ms/nnet_a/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.filt.sys
%WER 21.8 | 4459 42989 | 80.6 13.3 6.1 2.4 21.8 59.2 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_12/eval2000_hires.ctm.filt.sys
%WER 22.6 | 4459 42989 | 79.7 13.8 6.6 2.3 22.6 60.3 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_13/eval2000_hires.ctm.filt.sys
%WER 22.5 | 4459 42989 | 79.9 14.0 6.1 2.4 22.5 60.2 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.filt.sys
%WER 23.2 | 4459 42989 | 79.4 14.5 6.2 2.6 23.2 60.7 | exp/nnet2_online_ms/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_12/eval2000_hires.ctm.filt.sys

# Then this is the result obtained from this script.
# decode on SWBD only
%WER 14.6 | 1831 21395 | 87.0 8.7 4.4 1.6 14.6 52.4 | exp/nnet2_online_ms_p/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.swbd.filt.sys
%WER 15.0 | 1831 21395 | 86.5 8.8 4.6 1.6 15.0 52.8 | exp/nnet2_online_ms_p/nnet_a/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 14.5 | 1831 21395 | 87.1 8.7 4.3 1.6 14.5 52.3 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.swbd.filt.sys
%WER 15.2 | 1831 21395 | 86.4 9.0 4.6 1.6 15.2 52.8 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.0 | 1831 21395 | 86.6 8.8 4.6 1.6 15.0 52.4 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_tg/score_12/eval2000_hires.ctm.swbd.filt.sys
%WER 15.6 | 1831 21395 | 86.4 9.5 4.1 2.1 15.6 53.5 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_10/eval2000_hires.ctm.swbd.filt.sys

# decode on SWBD+CALLHM
%WER 21.1 | 4459 42989 | 81.1 12.9 6.0 2.2 21.1 58.7 | exp/nnet2_online_ms_p/nnet_a/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.filt.sys
%WER 21.7 | 4459 42989 | 80.6 13.4 6.0 2.3 21.7 59.3 | exp/nnet2_online_ms_p/nnet_a/decode_eval2000_hires_sw1_tg/score_11/eval2000_hires.ctm.filt.sys
%WER 21.0 | 4459 42989 | 81.2 12.9 6.0 2.2 21.0 58.4 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr/score_11/eval2000_hires.ctm.filt.sys
%WER 21.8 | 4459 42989 | 80.7 13.5 5.9 2.5 21.8 59.3 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_fsh_tgpr_per_utt/score_11/eval2000_hires.ctm.filt.sys
%WER 21.6 | 4459 42989 | 80.7 13.3 6.0 2.3 21.6 58.9 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_tg/score_11/eval2000_hires.ctm.filt.sys
%WER 22.4 | 4459 42989 | 80.4 14.0 5.6 2.8 22.4 60.3 | exp/nnet2_online_ms_p/nnet_a_online/decode_eval2000_hires_sw1_tg_per_utt/score_10/eval2000_hires.ctm.filt.sys


