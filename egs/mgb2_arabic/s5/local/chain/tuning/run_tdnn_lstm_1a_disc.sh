#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the alignment and the lattice generation/egs-dumping takes quite a bit
# of CPU time.


# Below is with 0.00002 and last_layer_factor=0.5
# this is the setting we're leaving in the script, but the discriminative training
# is not really helping.  Maybe we should try the frame-shifted version.
# steps/info/nnet3_disc_dir_info.pl exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbroutslow2
# exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbroutslow2:num-jobs=4;effective-lrate=2e-05;last-layer-factor=0.50;iters-per-epoch=138;epoch[0,1,2]:train-objf=[0.94,0.96,0.97],valid-objf=[0.95,0.96,0.96],train-counts=[0.24,0.12,0.10],valid-counts=[0.28,0.20,0.17]
# b01:s5_r2: steps/info/nnet3_disc_dir_info.pl exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbroutslow

# local/chain/compare_wer_general.sh --looped exp/chain_cleaned/tdnn_lstm1e_sp_bi exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbroutslow2:{1,2}
# System                tdnn_lstm1e_sp_bi tdnn_lstm1e_sp_bi_smbroutslow2:1 tdnn_lstm1e_sp_bi_smbroutslow2:2
# WER on dev(orig)            9.0       8.9       8.9
#         [looped:]           9.0       8.9       8.9
# WER on dev(rescored)        8.4       8.3       8.4
#         [looped:]           8.4       8.3       8.4
# WER on test(orig)           8.8       8.7       8.8
#         [looped:]           8.8       8.8       8.8
# WER on test(rescored)       8.4       8.3       8.4
#         [looped:]           8.3       8.4       8.5



# Below is with 0.00002 and last_layer_factor=1.0.
# b01:s5_r2: steps/info/nnet3_disc_dir_info.pl exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbr
# exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbr:num-jobs=4;lrate=2e-05;iters-per-epoch=138;epoch[0,1,2]:train-objf=[0.94,0.96,0.97],valid-objf=[0.95,0.96,0.96],train-counts=[0.24,0.12,0.09],valid-counts=[0.28,0.19,0.16]
# local/chain/compare_wer_general.sh --looped exp/chain_cleaned/tdnn_lstm1e_sp_bi exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbr:{1,2}
# System                tdnn_lstm1e_sp_bi tdnn_lstm1e_sp_bi_smbr:1 tdnn_lstm1e_sp_bi_smbr:2
# WER on dev(orig)            9.0       8.8       8.9
#         [looped:]           9.0       8.9       8.9
# WER on dev(rescored)        8.4       8.3       8.4
#         [looped:]           8.4       8.3       8.4
# WER on test(orig)           8.8       8.8       8.9
#         [looped:]           8.8       8.8       8.9
# WER on test(rescored)       8.4       8.4       8.5
#         [looped:]           8.3       8.4       8.5


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
# you can set disc_affix if you run different configurations, e.g. --disc-affix "_b"
disc_affix=



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


srcdir=exp/chain_mer80/tdnn_lstm_1a_sp_bi
graph_dir=$srcdir/graph
train_data_dir=data/train_mer80_sp_hires_comb
online_ivector_dir=exp/nnet3_mer80/ivectors_train_mer80_sp_hires_comb

## Objective options
criterion=smbr
one_silence_class=true

dir=${srcdir}_${criterion}${disc_affix}

## Egs options.  Give quite a few choices of chunk length,
## so it can split utterances without much gap or overlap.
frames_per_eg=300,280,150,120,100
frames_overlap_per_eg=0
frames_per_chunk_egs=200  # frames-per-chunk for decoding in alignment and
                          # denlat decoding.
frames_per_chunk_decoding=150  # frames-per-chunk for decoding when we test
                               # the models.
## these context options should match the training condition. (chunk_left_context,
## chunk_right_context)
## We set --extra-left-context-initial 0 and --extra-right-context-final 0
## directly in the script below, but this should also match the training condition.
extra_left_context=50
extra_right_context=0



## Nnet training options
effective_learning_rate=0.00002
max_param_change=1
num_jobs_nnet=4
num_epochs=2
regularization_opts=          # Applicable for providing --xent-regularize and --l2-regularize options,
                              # in chain models.
last_layer_factor=0.5    # have the output layer train slower than the others.. this can
                         # be helpful.
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
    --scale-opts "--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0" \
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
      --self-loop-scale 1.0 --acwt 1.0 \
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
    --criterion $criterion --drop-frames true  --acoustic-scale 1.0 \
    --num-epochs $num_epochs --one-silence-class $one_silence_class --minibatch-size "$minibatch_size" \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
    --last-layer-factor $last_layer_factor \
    --regularization-opts "$regularization_opts" \
    ${degs_dir} $dir
fi

if [ $stage -le 4 ]; then
  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in dev_non_overlap; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      if [ $num_jobs -gt 40 ]; then
        num_jobs=40
      fi
      iter=epoch$x
      # We don't test the iter "epoch${x}_adj", although it's computed,
      # because prior-adjustment doesn't make sense for chain models
      # and it degrades the results.

      (
        steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
         --acwt 1.0 --post-decode-acwt 10.0 \
         --frames-per-chunk "$frames_per_chunk_decoding" \
         --extra-left-context $extra_left_context \
         --extra-right-context $extra_right_context \
         --extra-left-context-initial 0 --extra-right-context-final 0 \
         --online-ivector-dir exp/nnet3_mer80/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${iter} || exit 1;

        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_test data/lang_test_fg data/${decode_set}_hires \
          $dir/decode_${decode_set}_${iter} \
          $dir/decode_${decode_set}_${iter}_fg || exit 1;
      ) &
    done
  done
fi

if [ $stage -le 5 ]; then
  # 'looped' decoding.  we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true

  for x in `seq $decode_start_epoch $num_epochs`; do
    for decode_set in dev_non_overlap; do
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      if [ $num_jobs -gt 40 ]; then
        num_jobs=40
      fi
      iter=epoch$x
      # We don't test the iter "epoch${x}_adj", although it's computed,
      # because prior-adjustment doesn't make sense for chain models
      # and it degrades the results.
      (
        steps/nnet3/decode_looped.sh \
          --nj $num_jobs --cmd "$decode_cmd" --iter $iter \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk 30 \
          --online-ivector-dir exp/nnet3_mer80/ivectors_${decode_set} \
          --scoring-opts "--min-lmwt 5 " \
          $graph_dir data/${decode_set}_hires $dir/decode_looped_${decode_set}_${iter} || exit 1;
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_test data/lang_test_fg \
          data/${decode_set}_hires \
          ${dir}/decode_looped_${decode_set}_${iter} ${dir}/decode_looped_${decode_set}_${iter}_fg || exit 1
      ) || touch $dir/.error &
    done
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi



wait;

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  # actually, keep the alignments in case we need them later.. they're slow to
  # create, and quite big.
  # rm ${srcdir}_ali/ali.*.gz || true

  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi


exit 0;

