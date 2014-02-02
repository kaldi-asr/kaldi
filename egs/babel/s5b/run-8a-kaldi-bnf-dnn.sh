#!/bin/bash

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

echo ---------------------------------------------------------------------
echo "Starting training the bottleneck network"
echo ---------------------------------------------------------------------

if [ ! -f exp_bnf/tri6_bnf/.done ]; then
  local/nnet2/train_tanh_bottleneck.sh \
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


if [ ! -f data_bnf/train_bnf/.done ]; then
  mkdir -p data_bnf
  # put the archives in mfcc/.
  local/nnet2/dump_bottleneck_features.sh --nj $train_nj --cmd "$train_cmd" \
    --transform-dir exp/tri5 data/train data_bnf/train_bnf exp_bnf/tri6_bnf exp/tri5_ali mfcc exp_bnf/dump_bnf
  touch data_bnf/train_bnf/.done
fi 

echo ---------------------------------------------------------------------
echo "Starting hybrid system building (over bottleneck features)"
echo ---------------------------------------------------------------------

# Wait till the main run-8b-kaldi-bnf-sgmm.sh gets to the stage where's it's
# finished aligning the tri6 model.      
echo "Waiting till exp_BNF/tri6_ali/.done exists...."   
while [ ! -f exp_BNF/tri6_ali/.done ]; do sleep 30; done     
  echo "...done waiting for exp_BNF/tri6_ali/.done"       

if [ ! -f exp/tri_bnf/.done ]; then
  steps/nnet2/train_pnorm.sh \
    --stage $dnn_train_stage --num-jobs-nnet $dnn_num_jobs \
    --num-threads $dnn_num_threads --mix-up $dnn_mixup \
    --minibatch-size $dnn_minibatch_size \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --max-change $dnn_max_change \
    --egs-opts "--feat-type raw" --lda-opts "--feat-type raw" --splice-width 5 \
    "${dnn_gpu_parallel_opts[@]}" --cmd "$train_cmd" \
      data/train_bnf data/lang exp_BNF/tri6_ali exp/tri_bnf || exit 1 

  touch exp/tri_bnf/.done 
fi

echo ---------------------------------------------------------------------
echo "Starting decoding the final system"
echo ---------------------------------------------------------------------

if [ -f exp/tri_bnf/.done ]; then
  datadir=data/dev10h
  decode=exp/tri_bnf/decode_dev10h
  if [ ! -f $decode/.done ]; then 
    local/nnet2/bnf_decode.sh --cmd "$decode_cmd" --nj $dev10h_nj \
      --beam $dnn_beam --stage $bnfdecode_stage --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dirid} \
      exp_BNF/tri6/graph ${datadir} $decode | tee $decode/decode.log

   shadow_set_extra_opts=
   local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
     --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt --wip $wip \
     "${shadow_set_extra_opts[@]}" ${datadir} data/lang ${decode}

    touch $decode/.done
  fi
fi
