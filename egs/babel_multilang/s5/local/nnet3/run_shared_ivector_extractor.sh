#!/bin/bash

. ./cmd.sh
set -e
stage=4
generate_alignments=true # false if doing ctc training
use_flp=false
speed_perturb=true
pitch_conf=conf/pitch.conf
num_ubm_gauss=512
num_ubm_frames=500000
global_extractor=exp/multi/nnet3    # The global ivector extractor dir
multi_data_dir=data/multi/train_sp_hires

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1;

. conf/common_vars.sh || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh


L=$1

. local/prepare_lang_conf.sh --fullLP $use_flp $L

if $use_flp; then
. local/prepare_flp_langconf.sh $L
else
. local/prepare_llp_langconf.sh $L
fi

langconf=langconf/$L/lang.conf

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. $langconf || exit 1;
train_set=train

if [ "$speed_perturb" == "true" ]; then 
  train_set=train_sp 
fi

if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  mkdir -p exp/$L/nnet3
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    --boost-silence $boost_sil \
    $numLeavesMLLT $numGaussMLLT data/$L/${train_set}_hires \
    data/$L/lang exp/$L/tri5_ali_sp exp/$L/nnet3/tri3b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 200 --num-frames $num_ubm_frames \
    $multi_data_dir $num_ubm_gauss exp/$L/nnet3/tri3b $global_extractor/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 200 \
    $multi_data_dir  $global_extractor/diag_ubm $global_extractor/extractor || exit 1;
fi

exit 0;
