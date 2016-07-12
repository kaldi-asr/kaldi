#!/bin/bash
# -v3 is as -v2 but it just consists 10 closest fullLP langs to GRG + GRG.
# -v2 is as 2c-bnf.sh but it consists of FullLP GRG.

# Copyright 2016  Pegah Ghahremani
# Apache 2.0

#This yields approx 70 hours of data
# this script generates bottleneck features from multilingual model
# trained on list of languages and dump the bnf for specific language L.
set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
. conf/common_vars.sh || exit 1;

set -u           #Fail on an undefined variable
skip_kws=true
skip_stt=false
bnf_train_stage=-100
stage=1
num_archives=20
bnf_init_lrate=0.0017
bnf_final_lrate=0.00017
bnf_layer=5
relu_dim=600
bottleneck_dim=42
speed_perturb=true
multidir=exp/nnet3/multi_bnf_10_close_lang_plus_grg
global_extractor=exp/multi/nnet3/extractor
lang_list=(GRG LIT MONG TUR KAZ KUR PSH SWA TOK IGBO DHO)
use_flp=true

. ./utils/parse_options.sh


L=$1

if $use_flp; then
. local/prepare_flp_langconf.sh $L
else
. local/prepare_llp_langconf.sh $L
fi

langconf=langconf/$L/lang.conf
[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1;
. $langconf || exit 1;

suffix=
if $speed_perturb; then
  suffix=_sp
fi
exp_dir=exp/$L
datadir=data/$L/train${suffix}_hires_mfcc_pitch
appended_dir=data/$L/train${suffix}_hires_mfcc_pitch_bnf
data_bnf_dir=data/$L/train${suffix}_bnf
dump_bnf_dir=bnf/$L
ivector_dir=$exp_dir/nnet3/ivectors_train${suffix}_gb
###############################################################################
#
# Training multilingual model with bottleneck layer
#
###############################################################################
mkdir -p $multidir${suffix}

if [ ! -f $multidir${suffix}/.done ]; then 
 echo "$0: Train Multilingual Bottleneck network using lang list = ${lang_list[@]}"
 ./local/nnet3/run_tdnn_joint_babel_sp_bnf.sh --dir $multidir \
    --avg-num-archives $num_archives \
    --global-extractor $global_extractor \
    --init-lrate $bnf_init_lrate \
    --final-lrate $bnf_final_lrate \
    --print-interval 200 \
    --train-stage $bnf_train_stage --stage $stage \
    --bottleneck-dim $bottleneck_dim --bnf-layer $bnf_layer --relu-dim $relu_dim  || exit 1;

  touch $multidir${suffix}/.done
fi
multidir=$multidir${suffix}

[ ! -d $dump_bnf_dir ] && mkdir -p $dump_bnf_dir
if [ ! -f $data_bnf_dir/.done ]; then
  mkdir -p $dump_bnf_dir
  # put the archives in ${dump_bnf_dir}/.
  steps/nnet3/dump_bottleneck_features.sh --use-gpu true --nj $train_nj --cmd "$train_cmd" \
    --ivector-dir $ivector_dir \
    --bnf-name renorm${bnf_layer} --feat-type raw \
    $datadir $data_bnf_dir \
    $multidir $dump_bnf_dir $exp_dir/make_train_bnf || exit 1; 
  touch $data_bnf_dir/.done
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
    $numLeavesMLLT $numGaussMLLT $appended_dir data/$L/lang $exp_dir/tri5_ali_sp $exp_dir/tri5b ;
  touch $exp_dir/tri5b/.done
fi

if [ ! $exp_dir/tri6/.done -nt $exp_dir/tri5b/.done ]; then
  steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT $appended_dir data/$L/lang \
    $exp_dir/tri5b $exp_dir/tri6
  touch $exp_dir/tri6/.done
fi

echo ---------------------------------------------------------------------
echo "$0: next, run run-6-bnf-sgmm-semisupervised.sh"
echo ---------------------------------------------------------------------

exit 0;
