#!/bin/bash

set -o pipefail
set -e

# This script does discriminative training on top of chain nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.

## dev set:
## %WER 14.1 | 507 17792 | 88.6 7.3 4.1 2.7 14.1 92.9 | 0.075 | exp/chain/tdnn/decode_dev/score_10_0.5/ctm.filt.filt.sys

## %WER 14.1 | 507 17792 | 89.3 7.7 2.9 3.5 14.1 94.5 | 0.125 | exp/chain/tdnn_sp_smbr/decode_dev_epoch1/score_10_0.5/ctm.filt.filt.sys
## %WER 13.9 | 507 17792 | 89.1 7.5 3.3 3.0 13.9 94.1 | 0.114 | exp/chain/tdnn_sp_smbr/decode_dev_epoch2/score_10_1.0/ctm.filt.filt.sys
## %WER 13.8 | 507 17792 | 89.2 7.5 3.2 3.0 13.8 93.7 | 0.095 | exp/chain/tdnn_sp_smbr/decode_dev_epoch3/score_10_1.0/ctm.filt.filt.sys
## %WER 13.8 | 507 17792 | 89.3 7.5 3.2 3.0 13.8 93.5 | 0.094 | exp/chain/tdnn_sp_smbr/decode_dev_epoch4/score_10_1.0/ctm.filt.filt.sys

## %WER 13.3 | 507 17792 | 89.7 6.9 3.4 2.9 13.3 92.1 | 0.000 | exp/chain/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys

## %WER 13.1 | 507 17792 | 90.1 7.0 3.0 3.1 13.1 92.1 | -0.004 | exp/chain/tdnn_sp_smbr/decode_dev_epoch1_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 13.0 | 507 17792 | 90.7 7.1 2.2 3.7 13.0 93.1 | 0.001 | exp/chain/tdnn_sp_smbr/decode_dev_epoch2_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 12.8 | 507 17792 | 90.3 6.9 2.7 3.2 12.8 92.3 | -0.009 | exp/chain/tdnn_sp_smbr/decode_dev_epoch3_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 12.8 | 507 17792 | 90.4 6.9 2.7 3.2 12.8 92.3 | -0.012 | exp/chain/tdnn_sp_smbr/decode_dev_epoch4_rescore/score_10_0.5/ctm.filt.filt.sys

## test set:
## %WER 13.8 | 1155 27512 | 89.4 7.5 3.1 3.2 13.8 87.9 | 0.101 | exp/chain/tdnn/decode_test/score_10_0.0/ctm.filt.filt.sys

## %WER 14.0 | 1155 27512 | 89.5 7.6 2.8 3.5 14.0 90.6 | 0.118 | exp/chain/tdnn_sp_smbr/decode_test_epoch1/score_10_0.5/ctm.filt.filt.sys
## %WER 14.0 | 1155 27512 | 89.6 7.6 2.8 3.6 14.0 91.1 | 0.115 | exp/chain/tdnn_sp_smbr/decode_test_epoch2/score_10_0.5/ctm.filt.filt.sys
## %WER 14.0 | 1155 27512 | 89.7 7.6 2.7 3.6 14.0 90.9 | 0.122 | exp/chain/tdnn_sp_smbr/decode_test_epoch3/score_10_0.5/ctm.filt.filt.sys
## %WER 14.0 | 1155 27512 | 89.7 7.6 2.7 3.7 14.0 91.0 | 0.123 | exp/chain/tdnn_sp_smbr/decode_test_epoch4/score_10_0.5/ctm.filt.filt.sys

## %WER 12.9 | 1155 27512 | 90.1 6.6 3.3 2.9 12.9 86.1 | 0.043 | exp/chain/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

## %WER 13.3 | 1155 27512 | 90.6 6.8 2.6 3.9 13.3 91.5 | 0.048 | exp/chain/tdnn_sp_smbr/decode_test_epoch1_rescore/score_10_0.0/ctm.filt.filt.sys
## %WER 13.1 | 1155 27512 | 90.4 6.7 2.9 3.5 13.1 90.3 | 0.042 | exp/chain/tdnn_sp_smbr/decode_test_epoch2_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 13.1 | 1155 27512 | 90.4 6.7 2.8 3.5 13.1 90.1 | 0.051 | exp/chain/tdnn_sp_smbr/decode_test_epoch3_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 13.0 | 1155 27512 | 90.1 6.6 3.4 3.1 13.0 88.7 | 0.030 | exp/chain/tdnn_sp_smbr/decode_test_epoch4_rescore/score_10_1.0/ctm.filt.filt.sys

## Cleanup Results:
## local/run_cleanup_segmentation.sh --cleanup-affix cleaned_b --pad-length 5 --max-incorrect-words 0 

## dev set:
## %WER 13.0 | 507 17792 | 89.1 7.5 3.4 2.1 13.0 86.4 | 0.026 | exp/chain_cleaned_b/tdnn/decode_dev/score_9_1.0/ctm.filt.filt.sys

## %WER 12.9 | 507 17792 | 89.6 7.5 2.9 2.5 12.9 86.4 | 0.057 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch1/score_10_0.5/ctm.filt.filt.sys
## %WER 12.7 | 507 17792 | 89.4 7.4 3.2 2.1 12.7 86.2 | 0.057 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch2/score_10_1.0/ctm.filt.filt.sys
## %WER 12.7 | 507 17792 | 89.5 7.4 3.1 2.1 12.7 86.0 | 0.054 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch3/score_10_1.0/ctm.filt.filt.sys
## %WER 12.7 | 507 17792 | 89.5 7.5 3.0 2.1 12.7 86.0 | 0.048 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch4/score_10_1.0/ctm.filt.filt.sys

## %WER 12.3 | 507 17792 | 90.3 6.8 2.9 2.5 12.3 86.4 | -0.015 | exp/chain_cleaned_b/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys

## %WER 11.9 | 507 17792 | 90.4 6.7 2.9 2.3 11.9 84.6 | -0.053 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch1_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.9 | 507 17792 | 90.5 6.8 2.7 2.4 11.9 84.0 | -0.054 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch2_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.7 | 507 17792 | 90.7 6.7 2.6 2.5 11.7 84.0 | -0.050 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch3_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.7 | 507 17792 | 90.7 6.8 2.5 2.5 11.7 84.2 | -0.052 | exp/chain_cleaned_b/tdnn_smbr/decode_dev_epoch4_rescore/score_10_0.5/ctm.filt.filt.sys

## test set:
## %WER 13.0 | 1155 27512 | 89.6 7.6 2.8 2.6 13.0 82.9 | 0.066 | exp/chain_cleaned_b/tdnn/decode_test/score_10_0.0/ctm.filt.filt.sys

## %WER 12.8 | 1155 27512 | 89.6 7.5 2.9 2.4 12.8 81.8 | 0.050 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch1/score_10_0.5/ctm.filt.filt.sys
## %WER 12.8 | 1155 27512 | 89.7 7.6 2.7 2.5 12.8 82.1 | 0.056 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch2/score_10_0.5/ctm.filt.filt.sys
## %WER 12.7 | 1155 27512 | 89.4 7.4 3.2 2.1 12.7 81.4 | 0.056 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch3/score_10_1.0/ctm.filt.filt.sys
## %WER 12.8 | 1155 27512 | 89.7 7.6 2.6 2.6 12.8 81.9 | 0.048 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch4/score_10_0.5/ctm.filt.filt.sys

## %WER 12.2 | 1155 27512 | 89.8 6.6 3.6 2.0 12.2 80.3 | -0.009 | exp/chain_cleaned_b/tdnn/decode_test_rescore/score_10_0.5/ctm.filt.filt.sys

## %WER 12.0 | 1155 27512 | 90.2 6.7 3.1 2.2 12.0 79.7 | -0.007 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch1_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.9 | 1155 27512 | 90.4 6.7 2.9 2.3 11.9 79.7 | -0.020 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch2_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.8 | 1155 27512 | 90.6 6.6 2.7 2.4 11.8 79.3 | -0.027 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch3_rescore/score_10_0.5/ctm.filt.filt.sys
## %WER 11.7 | 1155 27512 | 90.7 6.6 2.7 2.4 11.7 79.7 | -0.026 | exp/chain_cleaned_b/tdnn_smbr/decode_test_epoch4_rescore/score_10_0.5/ctm.filt.filt.sys

. cmd.sh

stage=0
train_stage=-10 # can be used to start training in the middle.
get_egs_stage=-10
use_gpu=true  # for training
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).

srcdir=exp/chain/tdnn_sp
train_data_dir=data/train_sp_hires
online_ivector_dir=exp/nnet3/ivectors_train_sp
degs_dir=                     # If provided, will skip the degs directory creation
lats_dir=                     # If provided, will skip denlats creation

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

## Objective options
criterion=smbr
one_silence_class=true

modify_learning_rates=false

dir=${srcdir}_${criterion}

## Egs options
frames_per_eg=150
frames_overlap_per_eg=30
truncate_deriv_weights=10

## Nnet training options
effective_learning_rate=0.000000125
max_param_change=1
num_jobs_nnet=4
num_epochs=4
regularization_opts="" #--xent-regularize=0.1 --l2-regularize=0.00005"          # Applicable for providing --xent-regularize and --l2-regularize options 
minibatch_size=64

## Decode options
decode_start_epoch=1 # can be used to avoid decoding all epochs, e.g. if we decided to run more.

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  num_threads=1
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
fi

if $modify_learning_rates; then
  dir=${dir}_modifylr
fi

if [ ! -f ${srcdir}/final.mdl ]; then
  echo "$0: expected ${srcdir}/final.mdl to exist; first run run_tdnn.sh or run_lstm.sh"
  exit 1;
fi

lang=data/lang

frame_subsampling_opt=
frame_subsampling_factor=1
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor)
  frame_subsampling_opt="--frame-subsampling-factor $(cat $srcdir/frame_subsampling_factor)"
fi

affix=    # Will be set if doing input frame shift
if [ $frame_subsampling_factor -ne 1 ]; then
  if [ $stage -le 0 ]; then
    mkdir -p ${online_ivector_dir}_fs
    cp -r $online_ivector_dir/{conf,ivector_period} ${online_ivector_dir}_fs

    rm ${online_ivector_dir}_fs/ivector_online.scp 2>/dev/null || true

    data_dirs=
    for x in `seq -$[frame_subsampling_factor/2] $[frame_subsampling_factor/2]`; do 
      steps/shift_feats.sh --cmd "$train_cmd --max-jobs-run 40" --nj 350 \
        $x $train_data_dir exp/shift_hires/ mfcc_hires
      utils/fix_data_dir.sh ${train_data_dir}_fs$x
      data_dirs="$data_dirs ${train_data_dir}_fs$x"
      awk -v nfs=$x '{print "fs"nfs"-"$0}' $online_ivector_dir/ivector_online.scp >> ${online_ivector_dir}_fs/ivector_online.scp
    done
    utils/combine_data.sh ${train_data_dir}_fs $data_dirs
    for x in `seq -$[frame_subsampling_factor/2] $[frame_subsampling_factor/2]`; do 
      rm -r ${train_data_dir}_fs$x
    done
  fi

  train_data_dir=${train_data_dir}_fs

  affix=_fs

  rm ${online_ivector_dir}_fs/ivector_online.scp 2>/dev/null || true
  for x in `seq -$[frame_subsampling_factor/2] $[frame_subsampling_factor/2]`; do 
    awk -v nfs=$x '{print "fs"nfs"-"$0}' $online_ivector_dir/ivector_online.scp >> ${online_ivector_dir}_fs/ivector_online.scp
  done
  online_ivector_dir=${online_ivector_dir}_fs
fi

if [ $stage -le 1 ]; then
  # get excellent GPU utilization though.]
  nj=350 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  steps/nnet3/align.sh  --cmd "$decode_cmd" --use-gpu false \
    --online-ivector-dir $online_ivector_dir \
    --scale-opts "--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0" \
    --nj $nj $train_data_dir $lang $srcdir ${srcdir}_ali${affix} ;
fi

if [ -z "$lats_dir" ]; then
  lats_dir=${srcdir}_denlats${affix}
  if [ $stage -le 2 ]; then
    nj=50  
    # this doesn't really affect anything strongly, except the num-jobs for one of
    # the phases of get_egs_discriminative.sh below.
    num_threads_denlats=6
    subsplit=40 # number of jobs that run per job (but 2 run at a time, so total jobs is 80, giving
    # total slots = 80 * 6 = 480.
    steps/nnet3/make_denlats.sh --cmd "$decode_cmd" \
      --self-loop-scale 1.0 --acwt 1.0 --determinize true \
      --online-ivector-dir $online_ivector_dir \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" \
      $train_data_dir $lang $srcdir ${lats_dir} ;
  fi
fi

model_left_context=`nnet3-am-info $srcdir/final.mdl | grep "left-context:" | awk '{print $2}'` 
model_right_context=`nnet3-am-info $srcdir/final.mdl | grep "right-context:" | awk '{print $2}'` 

left_context=$[model_left_context + extra_left_context]
right_context=$[model_right_context + extra_right_context]

valid_left_context=$[valid_left_context + frames_per_eg]
valid_right_context=$[valid_right_context + frames_per_eg]

cmvn_opts=`cat $srcdir/cmvn_opts` 

if [ -z "$degs_dir" ]; then
  degs_dir=${srcdir}_degs${affix}

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
      utils/create_split_dir.pl \
        /export/b{01,02,12,13}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
    fi
    # have a higher maximum num-jobs if
    if [ -d ${srcdir}_degs/storage ]; then max_jobs=10; else max_jobs=5; fi

    degs_opts="--determinize true --minimize true --remove-output-symbols true --remove-epsilons true --collapse-transition-ids true"

    steps/nnet3/get_egs_discriminative.sh \
      --cmd "$decode_cmd --max-jobs-run $max_jobs --mem 20G" --stage $get_egs_stage --cmvn-opts "$cmvn_opts" \
      --adjust-priors false --acwt 1.0 \
      --online-ivector-dir $online_ivector_dir \
      --left-context $left_context --right-context $right_context \
      --valid-left-context $valid_left_context --valid-right-context $valid_right_context \
      --priors-left-context $valid_left_context --priors-right-context $valid_right_context $frame_subsampling_opt \
      --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg ${degs_opts} \
      $train_data_dir $lang ${srcdir}_ali${affix} $lats_dir $srcdir/final.mdl $degs_dir ;
  fi
fi

if [ $stage -le 4 ]; then
  steps/nnet3/train_discriminative.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --effective-lrate $effective_learning_rate --max-param-change $max_param_change \
    --criterion $criterion --drop-frames true --acoustic-scale 1.0 \
    --num-epochs $num_epochs --one-silence-class $one_silence_class --minibatch-size $minibatch_size \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
    --regularization-opts "$regularization_opts" --use-frame-shift false \
    --truncate-deriv-weights $truncate_deriv_weights --adjust-priors false \
    --modify-learning-rates $modify_learning_rates \
      ${degs_dir} $dir ;
fi

graph_dir=$srcdir/graph
if [ $stage -le 5 ]; then
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      iter=epoch$x
      iter_opts="--iter $iter"

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
        --online-ivector-dir `dirname $online_ivector_dir`/ivectors_${decode_set} \
        --scoring-opts "--min_lmwt 5 --max_lmwt 15" \
        $graph_dir data/${decode_set}_hires \
        $dir/decode_${decode_set}${iter:+_$iter} || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test data/lang_rescore data/${decode_set}_hires \
        $dir/decode_${decode_set}${iter:+_$iter} \
        $dir/decode_${decode_set}${iter:+_$iter}_rescore || exit 1;
      ) &
    done
  done
fi
wait;

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  rm ${lats_dir}/lat.*.gz || true
  rm ${srcdir}_ali/ali.*.gz || true
  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi

exit 0;

