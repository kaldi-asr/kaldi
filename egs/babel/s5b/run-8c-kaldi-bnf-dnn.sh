#!/bin/bash

# This is really an alternative path to 8b, where we train a DNN instead of 
# an SGMM.

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

dnn_train_stage=-100

. utils/parse_options.sh

if [ ! -e exp_bnf/tri6/.done ]; then
  echo "$0: exp_bnf/tri6/.done does not exist"
  exit 1
fi


# We create an alignment with a lot of jobs, because the LDA accumulation
# when training the pnorm network will be slow, due to the large dimension.
if [ ! exp_bnf/tri6_ali_50/.done -nt exp_bnf/tri6/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Aligning fMLLR system with 50 jobs"
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data_bnf/train_app data/lang exp_bnf/tri6 exp_bnf/tri6_ali_50
  touch exp_bnf/tri6_ali_50/.done
fi


if [ ! exp_bnf/tri7_nnet/.done -nt exp_bnf/tri6_ali_50/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting hybrid system building (over bottleneck features)"
  echo ---------------------------------------------------------------------
  steps/nnet2/train_pnorm.sh \
    --stage $dnn_train_stage --mix-up $dnn_mixup \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --egs-opts "--feat-type raw" --lda-opts "--feat-type raw --lda-dim $dnn_output_dim" --splice-width 5 \
     "${dnn_gpu_parallel_opts[@]}" --cmd "$train_cmd" \
    data_bnf/train_app data/lang exp_bnf/tri6_ali_50 exp_bnf/tri7_nnet || exit 1 

  touch exp_bnf/tri7_nnet/.done 
fi


