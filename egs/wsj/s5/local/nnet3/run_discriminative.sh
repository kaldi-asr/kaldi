#!/bin/bash

set -e 
set -o pipefail

# this is run_discriminative.sh

# This script does discriminative training on top of nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
# 
# Note: rather than using any features we have dumped on disk, this script
# regenerates them from the wav data three times-- when we do lattice
# generation, numerator alignment and discriminative training.  This made the
# script easier to write and more generic, because we don't have to know where
# the features and the iVectors are, but of course it's a little inefficient.
# The time taken is dominated by the lattice generation anyway, so this isn't
# a huge deal.

. cmd.sh


stage=0
train_stage=-10
get_egs_stage=-10
use_gpu=true
srcdir=exp/nnet3/nnet_ms_a
criterion=smbr
drop_frames=false  # only matters for MMI.
frames_per_eg=150
frames_overlap_per_eg=30
effective_learning_rate=0.0000125
num_jobs_nnet=4
train_stage=-10 # can be used to start training in the middle.
decode_start_epoch=1 # can be used to avoid decoding all epochs, e.g. if we decided to run more.
num_epochs=4
degs_dir=
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).
lats_dir=
train_data_dir=data/train_si284_hires
online_ivector_dir=exp/nnet3/ivectors_train_si284
one_silence_class=true
truncate_deriv_weights=10
minibatch_size=64

adjust_priors=true

determinize=true
minimize=true
remove_output_symbols=true
remove_epsilons=true
collapse_transition_ids=true

modify_learning_rates=true
last_layer_factor=1.0

set -e
. cmd.sh
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


if [ -z "$lats_dir" ]; then
  lats_dir=${srcdir}_denlats
  if [ $stage -le 1 ]; then
    nj=50  # this doesn't really affect anything strongly, except the num-jobs for one of
    # the phases of get_egs_discriminative2.sh below.
    num_threads_denlats=6
    subsplit=40 # number of jobs that run per job (but 2 run at a time, so total jobs is 80, giving
    # total slots = 80 * 6 = 480.
    steps/nnet3/make_denlats.sh --cmd "$decode_cmd --mem 1G --num-threads $num_threads_denlats" \
      --online-ivector-dir $online_ivector_dir \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode_dnn.config \
      $train_data_dir data/lang $srcdir $lats_dir || exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=100 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  use_gpu=no
  gpu_opts=

  steps/nnet3/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
     --online-ivector-dir $online_ivector_dir \
     --nj $nj $train_data_dir data/lang $srcdir ${srcdir}_ali || exit 1;

  # the command below is a more generic, but slower, way to do it.
  # steps/online/nnet2/align.sh --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
  #    --nj $nj data/train_960 data/lang ${srcdir}_online ${srcdir}_ali || exit 1;
fi

left_context=14
right_context=10

cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1

if [ -z "$degs_dir" ]; then
  degs_dir=${srcdir}_degs_n${frames_per_eg}_o${frames_overlap_per_eg}_f
  if $determinize; then
    degs_dir=${degs_dir}d
  fi
  if $minimize; then
    degs_dir=${degs_dir}m
  fi
  if $remove_output_symbols; then
    degs_dir=${degs_dir}r
  fi
  if $remove_epsilons; then
    degs_dir=${degs_dir}e
  fi
  if $collapse_transition_ids; then
    degs_dir=${degs_dir}c
  fi

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
      utils/create_split_dir.pl \
        /export/b0{1,2,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
    fi
    # have a higher maximum num-jobs if
    if [ -d ${srcdir}_degs/storage ]; then max_jobs=10; else max_jobs=5; fi

    degs_opts="--determinize $determinize --minimize $minimize --remove-output-symbols $remove_output_symbols --remove-epsilons $remove_epsilons --collapse-transition-ids $collapse_transition_ids"

    steps/nnet3/get_egs_discriminative.sh \
      --cmd "$decode_cmd --max-jobs-run $max_jobs --mem 20G" --stage $get_egs_stage --cmvn-opts "$cmvn_opts" \
      --online-ivector-dir $online_ivector_dir --left-context $left_context --right-context $right_context \
      --criterion $criterion --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg ${degs_opts} \
      $train_data_dir data/lang ${srcdir}_ali $lats_dir $srcdir/final.mdl $degs_dir || exit 1;

    # the command below is a more generic, but slower, way to do it.
    #steps/online/nnet2/get_egs_discriminative2.sh \
      #  --cmd "$decode_cmd --max-jobs-run $max_jobs" \
      #  --criterion $criterion --drop-frames $drop_frames \
      #   data/train_960 data/lang ${srcdir}{_ali,_denlats,_online,_degs} || exit 1;
  fi
fi

d=`basename $degs_dir`
dir=${srcdir}_${criterion}_${effective_learning_rate}_degs${d##*degs}_ms${minibatch_size}

if $one_silence_class; then
  dir=${dir}_onesil
fi

if $modify_learning_rates; then
  dir=${dir}_modify
fi

if [ "$last_layer_factor" != "1.0" ]; then
  dir=${dir}_llf$last_layer_factor
fi

if [ $truncate_deriv_weights -ne 0 ]; then
  dir=${dir}_tr${truncate_deriv_weights}
fi

if [ $stage -le 4 ]; then
  bash -x steps/nnet3/train_discriminative.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --effective-lrate $effective_learning_rate \
    --criterion $criterion --drop-frames $drop_frames \
    --num-epochs $num_epochs --one-silence-class $one_silence_class --minibatch-size $minibatch_size \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
    --truncate-deriv-weights $truncate_deriv_weights --adjust-priors $adjust_priors \
    --modify-learning-rates $modify_learning_rates --last-layer-factor $last_layer_factor \
      ${degs_dir} $dir || exit 1;
fi

graph_dir=exp/tri4/graph_sw1_tg

if [ $stage -le 5 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for x in `seq $decode_start_epoch $num_epochs`; do
    iter=epoch$x.adj
    for lm_suffix in tgpr bd_tgpr; do
      graph_dir=exp/tri4b/graph_${lm_suffix}
      # use already-built graphs.
      for year in eval92 dev93; do
        steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_test_$year --iter $iter \
          $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year}_iter$iter || exit 1;
      done
    done
  done
fi

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  rm ${lats_dir}/lat.*.gz || true
  rm ${srcdir}_ali/ali.*.gz || true
  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi


exit 0;

