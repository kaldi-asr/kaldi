#!/bin/bash

# Note: there are two paths: the BNF+SGMM path is 8a->8b->8d
#                            the BNF+DNN path is 8a->8c->8e

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

skip_kws=true
skip_stt=false
bnf_train_stage=-100
dnn_train_stage=-100
bnfdecode_stage=-100


. utils/parse_options.sh

# Wait till the main run.sh gets to the stage where's it's
# finished aligning the tri5 model.      
echo "Waiting till exp/tri5_ali/.done exists...."   
while [ ! -f exp/tri5_ali/.done ]; do sleep 30; done     
  echo "...done waiting for exp/tri5_ali/.done"       

if [ ! -f exp_bnf/tri6_bnf/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting training the bottleneck network"
  echo ---------------------------------------------------------------------
  steps/nnet2/train_tanh_bottleneck.sh \
    --stage $bnf_train_stage --num-jobs-nnet $bnf_num_jobs \
    --num-threads $bnf_num_threads --mix-up $bnf_mixup \
    --minibatch-size $bnf_minibatch_size \
    --initial-learning-rate $bnf_init_learning_rate \
    --final-learning-rate $bnf_final_learning_rate \
    --num-hidden-layers $bnf_num_hidden_layers \
    --bottleneck-dim $bottleneck_dim --hidden-layer-dim $bnf_hidden_layer_dim \
    "${dnn_gpu_parallel_opts[@]}" --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri5_ali exp_bnf/tri6_bnf || exit 1 
  touch exp_bnf/tri6_bnf/.done
fi

[ ! -d param_bnf ] && mkdir -p param_bnf
if [ ! -f data_bnf/train_bnf/.done ]; then
  mkdir -p data_bnf
  # put the archives in plp/.
  steps/nnet2/dump_bottleneck_features.sh --nj $train_nj --cmd "$train_cmd" \
    --transform-dir exp/tri5 data/train data_bnf/train_bnf exp_bnf/tri6_bnf param_bnf exp_bnf/dump_bnf
  touch data_bnf/train_bnf/.done
fi 

if [ ! data_bnf/train/.done -nt data_bnf/train_bnf/.done ]; then
  steps/make_fmllr_feats.sh --cmd "$train_cmd -tc 10" \
    --nj $train_nj --transform-dir exp/tri5_ali  data_bnf/train_sat data/train \
    exp/tri5_ali exp_bnf/make_fmllr_feats/log param_bnf/ 

  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data_bnf/train_bnf data_bnf/train_sat data_bnf/train \
    exp_bnf/append_feats/log param_bnf/ 
  steps/compute_cmvn_stats.sh --fake data_bnf/train exp_bnf/make_fmllr_feats param_bnf
  rm -r data_bnf/train_sat

  touch data_bnf/train/.done
fi

if [ ! exp_bnf/tri5/.done -nt data_bnf/train/.done ]; then
  steps/train_lda_mllt.sh --splice-opts "--left-context=1 --right-context=1" \
    --dim 60 --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data_bnf/train data/lang exp/tri5_ali exp_bnf/tri5 ;
  touch exp_bnf/tri5/.done
fi

if [ ! exp_bnf/tri6/.done -nt exp_bnf/tri5/.done ]; then
  steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data_bnf/train data/lang exp_bnf/tri5 exp_bnf/tri6
  touch exp_bnf/tri6/.done
fi

echo ---------------------------------------------------------------------
echo "$0: next, run run-8b-kaldi-bnf-sgmm.sh or run-8c-kaldi-bnf-dnn.sh"
echo ---------------------------------------------------------------------

exit 0;


