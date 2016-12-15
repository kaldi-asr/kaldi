#!/bin/bash

echo "This script has not yet been tested, you would have to comment this statement if you want to run it. Please let us know if you see any issues" && exit 1;

set -o pipefail
set -e
# this is run_discriminative.sh

# This script does discriminative training on top of CE nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
# 


stage=0
train_stage=-10 # can be used to start training in the middle.
get_egs_stage=-10
use_gpu=true  # for training
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).
train_set=train_960_cleaned
gmm=tri6b_cleaned  # this is the source gmm-dir for the data-type of interest; it
                   # should have alignments for the specified training data.
nnet3_affix=_cleaned

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

srcdir=exp/nnet3${nnet3_affix}/tdnn_sp
gmm_dir=exp/${gmm}
graph_dir=$gmm_dir/graph_tgsmall
train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
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
effective_learning_rate=0.00000125
max_param_change=1
num_jobs_nnet=4
num_epochs=4
regularization_opts=          # Applicable for providing --xent-regularize and --l2-regularize options 
minibatch_size=64
adjust_priors=true            # May need to be set to false 
                              # because it does not help in some setups
modify_learning_rates=true
last_layer_factor=0.1

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

if [ $stage -le 1 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=350 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  steps/nnet3/align.sh  --cmd "$decode_cmd" --use-gpu false \
    --online-ivector-dir $train_ivector_dir \
     --nj $nj $train_data_dir data/lang $srcdir ${srcdir}_ali ;

fi

if [ -z "$lats_dir" ]; then
  lats_dir=${srcdir}_denlats
  if [ $stage -le 2 ]; then
    nj=50  
    # this doesn't really affect anything strongly, except the num-jobs for one of
    # the phases of get_egs_discriminative.sh below.
    num_threads_denlats=6
    subsplit=40 # number of jobs that run per job (but 2 run at a time, so total jobs is 80, giving
    # total slots = 80 * 6 = 480.
    steps/nnet3/make_denlats.sh --cmd "$decode_cmd" --determinize true \
      --online-ivector-dir $train_ivector_dir \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.config \
      $train_data_dir data/lang $srcdir ${lats_dir} ;
  fi
fi

model_left_context=`nnet3-am-info $srcdir/final.mdl | grep "left-context:" | awk '{print $2}'` 
model_right_context=`nnet3-am-info $srcdir/final.mdl | grep "right-context:" | awk '{print $2}'` 

left_context=$[model_left_context + extra_left_context]
right_context=$[model_right_context + extra_right_context]

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_opt="--frame-subsampling-factor $(cat $srcdir/frame_subsampling_factor)"
fi

cmvn_opts=`cat $srcdir/cmvn_opts` 

if [ -z "$degs_dir" ]; then
  degs_dir=${srcdir}_degs

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
      utils/create_split_dir.pl \
        /export/b{01,02,12,13}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
    fi
    # have a higher maximum num-jobs if
    if [ -d ${srcdir}_degs/storage ]; then max_jobs=10; else max_jobs=5; fi

    degs_opts="--determinize true --minimize true --remove-output-symbols true --remove-epsilons true --collapse-transition-ids true"

    steps/nnet3/get_egs_discriminative.sh \
      --cmd "$decode_cmd --max-jobs-run $max_jobs --mem 20G" --stage $get_egs_stage --cmvn-opts "$cmvn_opts" \
      --adjust-priors $adjust_priors \
      --online-ivector-dir $train_ivector_dir \
      --left-context $left_context --right-context $right_context \
      $frame_subsampling_opt \
      --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg ${degs_opts} \
      $train_data_dir data/lang ${srcdir}_ali $lats_dir $srcdir/final.mdl $degs_dir ;
  fi
fi

if [ $stage -le 4 ]; then
  steps/nnet3/train_discriminative.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --effective-lrate $effective_learning_rate --max-param-change $max_param_change \
    --criterion $criterion --drop-frames true \
    --num-epochs $num_epochs --one-silence-class $one_silence_class --minibatch-size $minibatch_size \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
    --regularization-opts "$regularization_opts" \
    --truncate-deriv-weights $truncate_deriv_weights --adjust-priors $adjust_priors \
    --modify-learning-rates $modify_learning_rates --last-layer-factor $last_layer_factor \
    ${degs_dir} $dir 
fi

if [ $stage -le 5 ]; then
  rm $dir/.error 2>/dev/null || true
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in test_clean test_other dev_clean dev_other; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      iter=epoch$x.adj
      
      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
        $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_tgsmall_$iter || exit 1
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
        data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,tgmed}_$iter  || exit 1
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,tglarge}_$iter || exit 1
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,fglarge}_$iter || exit 1
      ) || touch $dir/.error &
    done
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  rm ${lats_dir}/lat.*.gz || true
  rm ${srcdir}_ali/ali.*.gz || true
  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi


exit 0;

