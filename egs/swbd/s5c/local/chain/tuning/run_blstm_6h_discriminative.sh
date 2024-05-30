#!/usr/bin/env bash

set -o pipefail
set -e
# this is run_discriminative.sh

# This script does discriminative training on top of chain nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
#
. ./cmd.sh


stage=0
train_stage=-10 # can be used to start training in the middle.
get_egs_stage=-10
use_gpu=true  # for training
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).

# Frame chunk options that will be used for blstm models.
frames_per_chunk=150
extra_left_context=40
extra_right_context=40
extra_left_context_initial=-1
extra_right_context_final=-1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

srcdir=exp/chain/blstm_6h_sp
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

[ ! -z "$frames_per_chunk" ] && context_opts="$context_opts --frames-per-chunk $frames_per_chunk"
[ ! -z "$extra_left_context" ] && context_opts="$context_opts --extra-left-context $extra_left_context"
[ ! -z "$extra_right_context" ] && context_opts="$context_opts --extra-right-context $extra_right_context"
[ ! -z "$extra_left_context_initial" ] && context_opts="$context_opts --extra-left-context-initial $extra_left_context_initial"
[ ! -z "$extra_right_context_final" ] && context_opts="$context_opts --extra-right-context-final $extra_right_context_final"

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
    --online-ivector-dir $online_ivector_dir $context_opts \
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
      --online-ivector-dir $online_ivector_dir $context_opts \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.config \
      $train_data_dir $lang $srcdir ${lats_dir} ;
  fi
fi

model_left_context=`nnet3-am-info $srcdir/final.mdl | grep "left-context:" | awk '{print $2}'`
model_right_context=`nnet3-am-info $srcdir/final.mdl | grep "right-context:" | awk '{print $2}'`

left_context=$[model_left_context + extra_left_context]
right_context=$[model_right_context + extra_right_context]

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

    steps/nnet3/get_egs_discriminative.sh \
      --cmd "$decode_cmd --max-jobs-run $max_jobs --mem 20G" --stage $get_egs_stage --cmvn-opts "$cmvn_opts" \
      --adjust-priors false --acwt 1.0 \
      --online-ivector-dir $online_ivector_dir \
      --left-context $left_context --right-context $right_context \
      $frame_subsampling_opt \
      --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg \
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
      ${degs_dir} $dir ;
fi

graph_dir=$srcdir/graph_sw1_tg
if [ $stage -le 5 ]; then
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in train_dev eval2000 rt03; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      iter=epoch$x_adj

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} $context_opts \
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
