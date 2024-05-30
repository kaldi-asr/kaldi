#!/usr/bin/env bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2014  Tom Ko
#           2014  Vijay Peddinti
# Apache 2.0

# This example script demonstrates how speed perturbation of the data helps the nnet training in the SWB setup.

. ./cmd.sh
set -e
stage=0
train_stage=-10
use_gpu=true
splice_indexes="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2"
common_egs_dir=
dir=exp/nnet2_online/nnet_ms_sp
has_fisher=true

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


# Run the common stages of training, including training the iVector extractor
local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 6 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_perturbed
  for x in train_sp; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_sp
fi

if [ $stage -le 7 ]; then
  #obtain the alignment of the perturbed data
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train_sp data/lang exp/tri3 exp/tri3_ali_sp || exit 1
fi

if [ $stage -le 8 ]; then
  #Now perturb the high resolution daa
  utils/perturb_data_dir_speed.sh 0.9 data/train_hires data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train_hires data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train_hires data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/train_hires_sp data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_perturbed
  for x in train_hires_sp; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 70 --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_hires/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_hires/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_hires_sp
fi

if [ $stage -le 9 ]; then
  # We extract iVectors on all the train data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires_sp data/train_hires_sp_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_hires_sp_max2 exp/nnet2_online/extractor exp/nnet2_online/ivectors_train_hires_sp2 || exit 1;
fi

if [ $stage -le 10 ]; then
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs 3 --num-jobs-initial 2 --num-jobs-final 12 \
    --num-hidden-layers 6 --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_hires_sp2 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --add-layers-period 1 \
    --mix-up 6000 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    data/train_hires_sp data/lang exp/tri3_ali_sp $dir  || exit 1;
fi

if [ $stage -le 11 ]; then
  # dump iVectors for the testing data.
  for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $num_jobs \
        data/${decode_set}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_${decode_set}_hires || exit 1;
  done
fi

if [ $stage -le 12 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=$dir/decode_${decode_set}
      steps/nnet2/decode.sh --nj $num_jobs --cmd "$decode_cmd" --config conf/decode.config \
        --online-ivector-dir exp/nnet2_online/ivectors_${decode_set}_hires \
        exp/tri3/graph data/${decode_set}_hires $decode_dir || exit 1;
      steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/${decode_set}_hires $decode_dir $decode_dir.rescore || exit 1
  done
fi


if [ $stage -le 13 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi
wait;

if [ $stage -le 14 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for decode_set in dev test; do
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}_online/decode_${decode_set}
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $num_jobs \
      exp/tri3/graph data/${decode_set}_hires $decode_dir || exit 1;
    steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/${decode_set}_hires $decode_dir $decode_dir.rescore || exit 1
  done
fi

if [ $stage -le 15 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true exp/tri3/graph data/${decode_set}_hires $decode_dir || exit 1;
      steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/${decode_set}_hires $decode_dir $decode_dir.rescore || exit 1
  done
fi

if [ $stage -le 16 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector (--online false)
  for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt_offline
      steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true --online false exp/tri3/graph data/${decode_set}_hires \
          $decode_dir || exit 1;
      steps/lmrescore_const_arpa.sh data/lang_test data/lang_rescore data/${decode_set}_hires $decode_dir $decode_dir.rescore || exit 1
  done
fi
wait;

if [ $stage -le 17 ]; then
  # prepare the build for distribution
  cat <<EOF >${dir}_online/sample_decode.sh
. ./cmd.sh
data_dir=\$1  # e.g. data/dev_hires (to be prepared by the user, see egs/tedlium/run.sh for examples)
model_dir=\$2 # e.g. exp/nnet2_online/nnet_ms_sp_online (provided in the distribution)

decode_dir=\$model_dir/\`basename \$data_dir\`
num_jobs=\`cat \$data_dir/spk2utt | wc -l\`
# note that the graph directory (exp/tri3/graph) is not provided in the distribution
steps/online/nnet2/decode.sh --cmd "\$decode_cmd" --nj \$num_jobs \
  exp/tri3/graph \$data_dir \$decode_dir ;
EOF
  chmod +x ${dir}_online/sample_decode.sh
  dist_file=tedlium_`basename $dir`.tgz
  utils/prepare_online_nnet_dist_build.sh --other-files ${dir}_online/sample_decode.sh data/lang ${dir}_online $dist_file
  echo "NOTE: If you would like to upload this build ($dist_file) to kaldi-asr.org please check the process at http://kaldi-asr.org/uploads.html"
fi

exit 0;
