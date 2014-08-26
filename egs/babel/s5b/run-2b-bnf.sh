#!/bin/bash

# Copyright 2014  Pegah Ghahremani
# Apache 2.0

#Run supervised and semisupervised BNF training
#This yields approx 70 hours of data

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

set -u           #Fail on an undefined variable
skip_kws=true
skip_stt=false
semisupervised=true
unsup_string="_semisup"
bnf_train_stage=-100
bnf_weight_threshold=0.35
ali_dir=
ali_model=exp/tri6b_nnet/
weights_dir=exp/best_path_weights/unsup.seg/decode_unsup.seg/

. ./utils/parse_options.sh

if [ $babel_type == "full" ] && $semisupervised; then
  echo "Error: Using unsupervised training for fullLP is meaningless, use semisupervised=false "
  exit 1
fi


if $semisupervised ; then
  egs_string="--egs-dir exp_bnf${unsup_string}/tri6_bnf/egs"
  dirid=unsup.seg
else
  unsup_string=""  #" ": supervised training, _semi_supervised: unsupervised BNF training
  egs_string=""
  dirid=train
fi

datadir=data/${dirid}
exp_dir=exp_bnf${unsup_string}
data_bnf_dir=data_bnf${unsup_string}
param_bnf_dir=param_bnf${unsup_string}

if [ -z $ali_dir ] ; then
  # If alignment directory is not done, use exp/tri6_nnet_ali as alignment 
  # directory
  ali_dir=exp/tri6_nnet_ali
fi

if [ ! -f $ali_dir/.done ]; then
  echo "$0: Aligning supervised training data in exp/tri6_nnet_ali"

  [ ! -f $ali_model/final.mdl ] && echo -e "$ali_model/final.mdl not found!\nRun run-6-nnet.sh first!" && exit 1
  steps/nnet2/align.sh  --cmd "$train_cmd" \
    --use-gpu no --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/lang $ali_model $ali_dir || exit 1
  touch $ali_dir/.done
fi

###############################################################################
#
# Semi-supervised BNF training
#
###############################################################################
mkdir -p $exp_dir/tri6_bnf  
if [ ! -f $exp_dir/tri6_bnf/.done ]; then    
  if $semisupervised ; then

    [ ! -d $datadir ] && echo "Error: $datadir is not available!" && exit 1;
    echo "$0: Generate examples using unsupervised data in $exp_dir/tri6_nnet"
    if [ ! -f $exp_dir/tri6_bnf/egs/.done ]; then
      local/nnet2/get_egs_semi_supervised.sh \
        --cmd "$train_cmd" \
        "${dnn_update_egs_opts[@]}" \
        --transform-dir-sup exp/tri5_ali \
        --transform-dir-unsup exp/tri5/decode_${dirid} \
        --weight-threshold $bnf_weight_threshold \
        data/train $datadir data/lang \
        $ali_dir $weights_dir $exp_dir/tri6_bnf || exit 1;
      touch $exp_dir/tri6_bnf/egs/.done
    fi
   
  fi  

 echo "$0: Train Bottleneck network"
  steps/nnet2/train_tanh_bottleneck.sh \
    --stage $bnf_train_stage --num-jobs-nnet $bnf_num_jobs \
    --num-threads $bnf_num_threads --mix-up $bnf_mixup \
    --minibatch-size $bnf_minibatch_size \
    --initial-learning-rate $bnf_init_learning_rate \
    --final-learning-rate $bnf_final_learning_rate \
    --num-hidden-layers $bnf_num_hidden_layers \
    --bottleneck-dim $bottleneck_dim --hidden-layer-dim $bnf_hidden_layer_dim \
    --cmd "$train_cmd" $egs_string  \
    "${dnn_gpu_parallel_opts[@]}" \
    data/train data/lang $ali_dir $exp_dir/tri6_bnf || exit 1

  touch $exp_dir/tri6_bnf/.done
fi

[ ! -d $param_bnf_dir ] && mkdir -p $param_bnf_dir
if [ ! -f $data_bnf_dir/train_bnf/.done ]; then
  mkdir -p $data_bnf_dir
  # put the archives in ${param_bnf_dir}/.
  steps/nnet2/dump_bottleneck_features.sh --nj $train_nj --cmd "$train_cmd" \
    --transform-dir exp/tri5 data/train $data_bnf_dir/train_bnf \
    $exp_dir/tri6_bnf $param_bnf_dir $exp_dir/dump_bnf
  touch $data_bnf_dir/train_bnf/.done
fi 

if [ ! $data_bnf_dir/train/.done -nt $data_bnf_dir/train_bnf/.done ]; then
  steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir exp/tri5_ali  $data_bnf_dir/train_sat data/train \
    exp/tri5_ali $exp_dir/make_fmllr_feats/log $param_bnf_dir  

  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    $data_bnf_dir/train_bnf $data_bnf_dir/train_sat $data_bnf_dir/train \
    $exp_dir/append_feats/log $param_bnf_dir/ 
  steps/compute_cmvn_stats.sh --fake $data_bnf_dir/train \
  $exp_dir/make_fmllr_feats $param_bnf_dir
  rm -r $data_bnf_dir/train_sat

  touch $data_bnf_dir/train/.done
fi

if [ ! $exp_dir/tri5/.done -nt $data_bnf_dir/train/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60 --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT $data_bnf_dir/train data/lang exp/tri5_ali $exp_dir/tri5 ;
  touch $exp_dir/tri5/.done
fi

if [ ! $exp_dir/tri6/.done -nt $exp_dir/tri5/.done ]; then
  steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT $data_bnf_dir/train data/lang \
    $exp_dir/tri5 $exp_dir/tri6
  touch $exp_dir/tri6/.done
fi

echo ---------------------------------------------------------------------
echo "$0: next, run run-6-bnf-sgmm-semisupervised.sh"
echo ---------------------------------------------------------------------

exit 0;
