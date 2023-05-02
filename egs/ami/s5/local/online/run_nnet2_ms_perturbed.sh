#!/usr/bin/env bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2014  Tom Ko
#           2014  Vijay Peddinti
# Apache 2.0

# This example script demonstrates how speed perturbation of the data helps the nnet training in the SWB setup.

. ./cmd.sh
set -e
stage=1
train_stage=-10
use_gpu=true
splice_indexes="layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3 layer3/-7:2 layer4/-3:3"
common_egs_dir=
has_fisher=true
mic=ihm
nj=70
affix=
hidden_dim=950
num_threads_ubm=32
use_sat_alignments=true
fix_nnet=false
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
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    parallel_opts="$parallel_opts --config conf/queue_no_k20.conf --allow-k20 false"
    # that config is like the default config in the text of queue.pl, but adding the following lines.
    # default allow_k20=true
    # option allow_k20=true
    # option allow_k20=false -l 'hostname=!g01&!g02&!b06'
    # It's a workaround for an NVidia CUDA library bug for our currently installed version
    # of the CUDA toolkit, that only shows up on k20's
  fi

  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

dir=exp/$mic/nnet2_online/nnet_ms_sp${affix:+_$affix}

if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/$mic/tri4a
  align_script=steps/align_fmllr.sh
else
  gmm_dir=exp/$mic/tri3a
  align_script=steps/align_si.sh
fi
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}

# Run the common stages of training, including training the iVector extractor
local/online/run_nnet2_common.sh --stage $stage --mic $mic \
  --use-sat-alignments $use_sat_alignments \
  --num-threads-ubm $num_threads_ubm|| exit 1;

if [ $stage -le 6 ]; then
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
fi

if [ $stage -le 7 ]; then
  $align_script --nj $nj --cmd "$train_cmd" \
    data/$mic/train_sp data/lang $gmm_dir ${gmm_dir}_sp_ali || exit 1
fi

if [ $stage -le 8 ]; then
  #Now perturb the high resolution daa
  utils/perturb_data_dir_speed.sh 0.9 data/$mic/train_hires data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/$mic/train_hires data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/$mic/train_hires data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/$mic/train_hires_sp data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_${mic}_perturbed
  for x in train_hires_sp; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf \
      data/$mic/$x exp/make_${mic}_hires/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$mic/$x exp/make_${mic}_hires/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/$mic/train_hires_sp
fi

if [ $stage -le 9 ]; then
  # We extract iVectors on all the train data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/$mic/train_hires_sp data/$mic/train_hires_sp_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/$mic/train_hires_sp_max2 exp/$mic/nnet2_online/extractor exp/$mic/nnet2_online/ivectors_train_hires_sp2 || exit 1;
fi

if [ $stage -le 10 ]; then
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs 3 --num-jobs-initial 2 --num-jobs-final 12 \
    --num-hidden-layers 6 --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/$mic/nnet2_online/ivectors_train_hires_sp2 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --add-layers-period 1 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim $hidden_dim \
    --pnorm-output-dim $hidden_dim \
    --fix-nnet $fix_nnet \
    data/$mic/train_hires_sp data/lang ${gmm_dir}_sp_ali $dir  || exit 1;
fi

if [ $stage -le 11 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/$mic/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi
wait;

if [ $stage -le 12 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for decode_set in dev eval; do
    (
    num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}_online/decode_${decode_set}
    steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
      $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
    ) &
  done
fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt
      steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi

if [ $stage -le 14 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector (--online false)
  for decode_set in dev eval; do
    (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}_online/decode_${decode_set}_utt_offline
      steps/online/nnet2/decode.sh --config conf/decode.conf --cmd "$decode_cmd" --nj $num_jobs \
        --per-utt true --online false $graph_dir data/$mic/${decode_set}_hires \
          $decode_dir || exit 1;
    ) &
  done
fi
wait;

exit 0;
