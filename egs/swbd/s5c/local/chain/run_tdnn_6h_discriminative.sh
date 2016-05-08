#!/bin/bash

set -o pipefail
set -e
# this is run_discriminative.sh

# This script does discriminative training on top of chain nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
#
# eval2000

# chain 7b
# %WER 17.2 | 4459 42989 | 84.8 10.2 5.0 2.0 17.2 54.4 | exp/chain/tdnn_7b_sp/decode_eval2000_sw1_tg/score_10_0.0/eval2000_hires.ctm.filt.sys

# chain 7b + smbr
# %WER 16.9 | 4459 42989 | 85.2 10.3 4.5 2.1 16.9 54.4 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_tg_epoch1/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 16.9 | 4459 42989 | 85.4 10.5 4.1 2.3 16.9 54.2 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_tg_epoch2/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 17.0 | 4459 42989 | 85.3 10.4 4.3 2.3 17.0 54.5 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_tg_epoch3/score_12_0.0/eval2000_hires.ctm.filt.sys
# %WER 17.1 | 4459 42989 | 85.2 10.5 4.3 2.4 17.1 54.5 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_tg_epoch4/score_12_0.5/eval2000_hires.ctm.filt.sys

# chain 7b
# %WER 15.5 | 4459 42989 | 86.3 9.0 4.7 1.8 15.5 51.3 | exp/chain/tdnn_7b_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.filt.sys

# chain 7b + smbr
# %WER 15.2 | 4459 42989 | 86.8 9.1 4.1 2.0 15.2 51.2 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_fsh_fg_epoch1/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 15.1 | 4459 42989 | 86.9 9.0 4.1 2.0 15.1 51.3 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_fsh_fg_epoch2/score_12_0.0/eval2000_hires.ctm.filt.sys
# %WER 15.1 | 4459 42989 | 87.0 9.1 3.9 2.1 15.1 51.2 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_fsh_fg_epoch3/score_12_0.5/eval2000_hires.ctm.filt.sys
# %WER 15.2 | 4459 42989 | 87.0 9.2 3.8 2.2 15.2 51.5 | exp/chain/tdnn_7b_sp_smbr/decode_eval2000_sw1_fsh_fg_epoch4/score_12_0.5/eval2000_hires.ctm.filt.sys


# RT'03

# chain 7b
# %WER 21.6 | 8420 76157 | 80.5 12.8 6.7 2.1 21.6 53.7 | exp/chain/tdnn_7b_sp/decode_rt03_sw1_tg/score_9_0.0/rt03_hires.ctm.filt.sys

# chain 7b + smbr
# %WER 21.0 | 8420 76157 | 81.3 12.8 5.8 2.4 21.0 53.0 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_tg_epoch1/score_10_0.0/rt03_hires.ctm.filt.sys
# %WER 20.8 | 8420 76157 | 81.6 12.5 6.0 2.4 20.8 53.0 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_tg_epoch2/score_11_0.0/rt03_hires.ctm.filt.sys
# %WER 20.8 | 8420 76157 | 81.6 12.6 5.8 2.5 20.8 53.1 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_tg_epoch3/score_11_0.5/rt03_hires.ctm.filt.sys
# %WER 20.9 | 8420 76157 | 81.7 12.7 5.6 2.6 20.9 53.2 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_tg_epoch4/score_11_0.0/rt03_hires.ctm.filt.sys

# chain 7b
# %WER 19.0 | 8420 76157 | 82.7 10.2 7.2 1.7 19.0 50.0 | exp/chain/tdnn_7b_sp/decode_rt03_sw1_fsh_fg/score_10_0.0/rt03_hires.ctm.filt.sys

# chain 7b + smbr
# %WER 18.2 | 8420 76157 | 83.7 10.4 5.9 1.9 18.2 49.3 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_fsh_fg_epoch1/score_11_0.0/rt03_hires.ctm.filt.sys
# %WER 18.1 | 8420 76157 | 83.9 10.7 5.4 2.1 18.1 49.3 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_fsh_fg_epoch2/score_11_0.0/rt03_hires.ctm.filt.sys
# %WER 18.1 | 8420 76157 | 84.0 10.7 5.3 2.1 18.1 49.4 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_fsh_fg_epoch3/score_11_1.0/rt03_hires.ctm.filt.sys
# %WER 18.2 | 8420 76157 | 83.8 10.5 5.7 2.1 18.2 49.6 | exp/chain/tdnn_7b_sp_smbr/decode_rt03_sw1_fsh_fg_epoch4/score_12_1.0/rt03_hires.ctm.filt.sys

. cmd.sh

stage=0
train_stage=-10 # can be used to start training in the middle.
get_egs_stage=-10
use_gpu=true  # for training
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

srcdir=exp/chain/tdnn_7b_sp
train_data_dir=data/train_nodup_sp_hires
online_ivector_dir=exp/nnet3/ivectors_train_nodup_sp
degs_dir=                     # If provided, will skip the degs directory creation
lats_dir=                     # If provided, will skip denlats creation

## Objective options
criterion=smbr
one_silence_class=true

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
regularization_opts="--xent-regularize=0.1 --l2-regularize=0.00005"          # Applicable for providing --xent-regularize and --l2-regularize options 
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
fi
    
rm ${online_ivector_dir}_fs/ivector_online.scp 2>/dev/null || true
for x in `seq -$[frame_subsampling_factor/2] $[frame_subsampling_factor/2]`; do 
  awk -v nfs=$x '{print "fs"nfs"-"$0}' $online_ivector_dir/ivector_online.scp >> ${online_ivector_dir}_fs/ivector_online.scp
done
online_ivector_dir=${online_ivector_dir}_fs

if [ $stage -le 1 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
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
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.config \
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
        /export/b0{1,2,12,13}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
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
    --modify-learning-rates false \
      ${degs_dir} $dir ;
fi

graph_dir=$srcdir/graph_sw1_tg
if [ $stage -le 5 ]; then
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in train_dev eval2000 rt03; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      iter=epoch$x.adj
      
      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_sw1_tg_$iter ;
      if $has_fisher; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
          $dir/decode_${decode_set}_sw1_{tg,fsh_fg}_$iter ;
      fi
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

