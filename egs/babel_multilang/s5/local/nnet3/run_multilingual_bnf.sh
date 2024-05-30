#!/usr/bin/env bash

# This script trains a multilingual model using 6 layer TDNN + Xent
# with 42 dim bottleneck layer in th fifth layer.
# Then it extracts bottleneck features for input language "lang" and
# train SAT model using these feautures.

# Copyright 2016  Pegah Ghahremani
# Apache 2.0

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
. conf/common_vars.sh || exit 1;

set -u           #Fail on an undefined variable
bnf_train_stage=-10 # the stage variable used in multilingual bottleneck training.
stage=1
speed_perturb=true
multilingual_dir=exp/nnet3/multi_bnf
global_extractor=exp/multi/nnet3/extractor
bnf_dim=42
. ./utils/parse_options.sh


lang=$1

langconf=conf/$lang/lang.conf

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1;
. $langconf || exit 1;

[ ! -f local.conf ] && echo 'the file local.conf does not exist!' && exit 1;
. local.conf || exit 1;

suffix=
if $speed_perturb; then
  suffix=_sp
fi

exp_dir=exp/$lang
datadir=data/$lang/train${suffix}_hires_mfcc_pitch
appended_dir=data/$lang/train${suffix}_hires_mfcc_pitch_bnf
data_bnf_dir=data/$lang/train${suffix}_bnf
dump_bnf_dir=bnf/$lang
ivector_dir=$exp_dir/nnet3/ivectors_train${suffix}_gb
###############################################################################
#
# Training multilingual model with bottleneck layer
#
###############################################################################
mkdir -p $multilingual_dir${suffix}

if [ ! -f $multilingual_dir${suffix}/.done ]; then
  echo "$0: Train multilingual DNN using Bottleneck layer with lang list = ${lang_list[@]}"
  . local/nnet3/run_tdnn_multilingual.sh --dir $multilingual_dir \
     --bnf-dim $bnf_dim \
     --global-extractor $global_extractor \
     --train-stage $bnf_train_stage --stage $stage  || exit 1;

  touch $multilingual_dir${suffix}/.done
else
  echo "$0 Skip multilingual DNN training; you can force to run this step by deleting $multilingual_dir${suffix}/.done"
fi

[ ! -d $dump_bnf_dir ] && mkdir -p $dump_bnf_dir
if [ ! -f $data_bnf_dir/.done ]; then
  multilingual_dir=$multilingual_dir${suffix}
  mkdir -p $dump_bnf_dir
  # put the archives in ${dump_bnf_dir}/.
  steps/nnet3/make_bottleneck_features.sh --use-gpu true --nj 70 --cmd "$train_cmd" \
    --ivector-dir $ivector_dir \
    tdnn_bn.renorm $datadir $data_bnf_dir \
    $multilingual_dir $dump_bnf_dir $exp_dir/make_train_bnf || exit 1;
  touch $data_bnf_dir/.done
else
  echo "$0 Skip Bottleneck feature extraction; You can force to run this step deleting $data_bnf_dir/.done."
fi

if [ ! -d $appended_dir/.done ]; then
  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    $data_bnf_dir $datadir $appended_dir \
    $exp_dir/append_hires_mfcc_bnf $dump_bnf_dir || exit 1;
  steps/compute_cmvn_stats.sh $appended_dir \
    $exp_dir/make_cmvn_mfcc_bnf $dump_bnf_dir || exit 1;
  touch $appended_dir/.done
fi

if [ ! $exp_dir/tri5b/.done -nt $data_bnf_dir/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60 --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT $appended_dir data/$lang/lang $exp_dir/tri5_ali_sp $exp_dir/tri5b ;
  touch $exp_dir/tri5b/.done
fi

if [ ! $exp_dir/tri6/.done -nt $exp_dir/tri5b/.done ]; then
  steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT $appended_dir data/$lang/lang \
    $exp_dir/tri5b $exp_dir/tri6
  touch $exp_dir/tri6/.done
fi

exit 0;
