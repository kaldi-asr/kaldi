#!/bin/bash

. ./cmd.sh
set -e
stage=4
suffix= # _sp, to use speed-perturbed data to generate lda+mllt model.

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1;

. conf/common_vars.sh || exit 1;

[ ! -f local.conf ] && echo 'the file local.conf does not exist!' && exit 1;
. local.conf || exit 1;

. ./utils/parse_options.sh

lda_mllt_lang=$1
multi_data_dir=$2
global_extractor_dir_dir=$3

langconf=conf/$lda_mllt_lang/lang.conf
[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. $langconf || exit 1;

if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  mkdir -p exp/$lda_mllt_lang/nnet3
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    --boost-silence $boost_sil \
    $numLeavesMLLT $numGaussMLLT data/$lda_mllt_lang/train${suffix}_hires \
    data/$lda_mllt_lang/lang exp/$lda_mllt_lang/tri5_ali${suffix} exp/$lda_mllt_lang/nnet3/tri3b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 200 --num-frames 500000 \
    $multi_data_dir $numGaussUBM exp/$lda_mllt_lang/nnet3/tri3b $global_extractor_dir_dir/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 200 \
    $multi_data_dir  $global_extractor_dir_dir/diag_ubm $global_extractor_dir_dir/extractor || exit 1;
fi
exit 0;
