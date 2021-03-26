#!/usr/bin/env bash

# This script does discriminative training on top of CE nnet3 system.  To
# simplify things, this assumes you are using the "cleaned" data (since this is
# generally better), i.e. it won't work if you used options to run_tdnn_lstm_1a.sh
# to use the non-cleaned data.
#
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the alignment and the lattice generation/egs-dumping takes quite a bit
# of CPU time.

# below is with the current settings (effective_learning_rate=0.0000025, last_layer_factor=0.5):
# steps/info/nnet3_disc_dir_info.pl exp/nnet3_cleaned/tdnn_lstm1a_sp_smbrslow
# exp/nnet3_cleaned/tdnn_lstm1a_sp_smbrslow:num-jobs=4;effective-lrate=2.5e-06;last-layer-factor=0.50;iters-per-epoch=55;epoch[0,1,2,3]:train-objf=[0.94,0.96,0.97,0.97],valid-objf=[0.91,0.93,0.93,0.93],train-counts=[0.40,0.25,0.17,0.12],valid-counts=[0.57,0.31,0.34,0.35]

# local/nnet3/compare_wer.sh --looped exp/nnet3_cleaned/tdnn_lstm1a_sp exp/nnet3_cleaned/tdnn_lstm1a_sp_smbrslow:{1,2,3}
# System                tdnn_lstm1a_sp tdnn_lstm1a_sp_smbrslow:1 tdnn_lstm1a_sp_smbrslow:2 tdnn_lstm1a_sp_smbrslow:3
# WER on dev(orig)           11.0       9.4       9.4       9.4
#         [looped:]          11.0       9.4       9.5       9.4
# WER on dev(rescored)       10.3       8.8       8.7       8.7
#         [looped:]          10.3       8.8       8.9       8.9
# WER on test(orig)          10.8       9.6       9.7       9.6
#         [looped:]          10.7       9.6       9.6       9.7
# WER on test(rescored)      10.1       9.1       9.2       9.1
#         [looped:]          10.0       9.1       9.2       9.1

# Below is with twice the lrate (5e-06) and the same last-layer-factor (0.5).  Trained too fast.
# exp/nnet3_cleaned/tdnn_lstm1a_sp_smbr:num-jobs=4;effective-lrate=5e-06;last-layer-factor=0.50;iters-per-epoch=55;epoch[0,1,2,3]:train-objf=[0.94,0.97,0.97,0.98],valid-objf=[0.91,0.93,0.93,0.93],train-counts=[0.40,0.22,0.12,0.09],valid-counts=[0.57,0.31,0.27,0.32]
#  I'm not showing the looped decoding results with this older step;
#  there was a script bug (now fixed) and I don't want to rerun them.
# local/nnet3/compare_wer.sh  exp/nnet3_cleaned/tdnn_lstm1a_sp exp/nnet3_cleaned/tdnn_lstm1a_sp_smbr:{1,2,3}
# System                tdnn_lstm1a_sp tdnn_lstm1a_sp_smbr:1 tdnn_lstm1a_sp_smbr:2 tdnn_lstm1a_sp_smbr:3
# WER on dev(orig)           11.0       9.4       9.4       9.5
# WER on dev(rescored)       10.3       8.8       8.8       8.9
# WER on test(orig)          10.8       9.6       9.8       9.8
# WER on test(rescored)      10.1       9.1       9.3       9.4

set -e
set -uo pipefail

stage=1
train_stage=-10 # can be used to start training in the middle.
get_egs_stage=0
use_gpu=true  # for training
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like
               # alignments and degs).
degs_dir=  # set this to use preexisting degs.
nj=400 # have a high number of jobs because this could take a while, and we might
       # have some stragglers.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

graph_dir=exp/tri3_cleaned/graph
srcdir=exp/nnet3_cleaned/tdnn_lstm1a_sp
train_data_dir=data/train_cleaned_sp_hires_comb
online_ivector_dir=exp/nnet3_cleaned/ivectors_train_cleaned_sp_hires_comb

## Objective options
criterion=smbr
one_silence_class=true

# originally ran with effective_learning_rate=0.000005,
# changing to effective_learning_rate=0.0000025 and using affix=slow

# you can set --disc-affix if you run different configurations.
disc_affix=

dir=${srcdir}_${criterion}${disc_affix}

## Egs options.  Give quite a few choices of chunk length,
## so it can split utterances without much gap or overlap.
frames_per_eg=300,280,150,120,100
frames_overlap_per_eg=0
frames_per_chunk_egs=200  # for alignments and denlat creation.
frames_per_chunk_decoding=50  # for decoding; should be the same as the value
                              # used in the script that trained the nnet.
                              # We didn't set the frames_per_chunk in
                              # run_tdnn_lstm_1a.sh, so it defaults to 50.
## these context options should match the training condition. (chunk_left_context,
## chunk_right_context)
## We set --extra-left-context-initial 0 and --extra-right-context-final 0
## directly in the script below, but this should also match the training condition.
## note: --extra-left-context should be the same as the chunk_left_context (or in
## general, the argument of --egs.chunk-left-context) in the baseline script.
extra_left_context=40
extra_right_context=0



## Nnet training options
effective_learning_rate=0.0000025
last_layer_factor=0.5
max_param_change=1
num_jobs_nnet=4
num_epochs=3
regularization_opts=          # Applicable for providing --xent-regularize and --l2-regularize options,
                              # in chain models.
minibatch_size="300=32,16/150=64,32"  # rule says: if chunk size is closer to 300, use minibatch size 32 (or 16 for mop-up);
                                      # if chunk size is closer to 150, use mini atch size of 64 (or 32 for mop-up).


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
  echo "$0: expected ${srcdir}/final.mdl to exist"
  exit 1;
fi

if [ $stage -le 1 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  steps/nnet3/align.sh  --cmd "$decode_cmd" --use-gpu false \
    --frames-per-chunk $frames_per_chunk_egs \
    --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --online-ivector-dir $online_ivector_dir \
    --nj $nj $train_data_dir data/lang $srcdir ${srcdir}_ali ;
fi


if [ -z "$degs_dir" ]; then

  if [ $stage -le 2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
      utils/create_split_dir.pl \
        /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/${srcdir}_degs/storage ${srcdir}_degs/storage
    fi
    if [ -d ${srcdir}_degs/storage ]; then max_copy_jobs=10; else max_copy_jobs=5; fi

    steps/nnet3/get_degs.sh \
      --cmd "$decode_cmd --mem 10G" --num-threads 3 \
      --max-copy-jobs $max_copy_jobs \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 --extra-right-context-final 0 \
      --frames-per-chunk-decoding "$frames_per_chunk_egs" \
      --stage $get_egs_stage \
      --online-ivector-dir $online_ivector_dir \
      --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg \
      $train_data_dir data/lang ${srcdir} ${srcdir}_ali ${srcdir}_degs || exit 1
  fi
fi

if [ $stage -le 3 ]; then
  [ -z "$degs_dir" ] && degs_dir=${srcdir}_degs
  steps/nnet3/train_discriminative.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --effective-lrate $effective_learning_rate --max-param-change $max_param_change \
    --last-layer-factor $last_layer_factor \
    --criterion $criterion --drop-frames true \
    --num-epochs $num_epochs --one-silence-class $one_silence_class --minibatch-size "$minibatch_size" \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
    --regularization-opts "$regularization_opts" \
    ${degs_dir} $dir
fi

if [ $stage -le 4 ]; then
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      for iter in epoch$x epoch${x}_adj; do
      (
        steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk "$frames_per_chunk_decoding" \
        --online-ivector-dir exp/nnet3_cleaned/ivectors_${decode_set}_hires \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${iter} || exit 1;

        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang data/lang_rescore data/${decode_set}_hires \
          $dir/decode_${decode_set}_${iter} \
          $dir/decode_${decode_set}_${iter}_rescore || exit 1;
      ) &
      done
    done
  done
fi
wait;

if [ $stage -le 5 ]; then
  # 'looped' decoding.  we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true

  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in dev test; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      iter=epoch$x
      # We don't test the iter "epoch${x}_adj", although it's computed,
      # because prior-adjustment doesn't make sense for chain models
      # and it degrades the results.
      (
        steps/nnet3/decode_looped.sh \
          --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
          --frames-per-chunk 30 \
          --online-ivector-dir exp/nnet3_cleaned/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
          $graph_dir data/${decode_set}_hires $dir/decode_looped_${decode_set}_${iter} || exit 1;
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
          data/${decode_set}_hires \
          ${dir}/decode_looped_${decode_set}_${iter} ${dir}/decode_looped_${decode_set}_${iter}_rescore || exit 1
      ) || touch $dir/.error &
    done
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi



if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  # actually, keep the alignments in case we need them later.. they're slow to
  # create, and quite big.
  # rm ${srcdir}_ali/ali.*.gz || true

  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi

exit 0;
