#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.
echo "Waiting till exp/tri5_ali/.done exists...."
while [ ! -f exp/tri5_ali/.done ]; do sleep 30; done
echo "...done waiting for exp/tri5_ali/.done"

# This parameter will be used when the training dies at a certain point.
train_stage=-100

if [ ! -f exp/tri6_nnet/.done ]; then
  steps/nnet2/train_pnorm.sh \
    --stage $train_stage --num-jobs-nnet $dnn_num_jobs \
    --num-threads $dnn_num_threads --mix-up $dnn_mixup \
    --minibatch-size $dnn_minibatch_size \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --max-change $dnn_max_change \
    --parallel-opts "$dnn_parallel_opts" --cmd "$train_cmd" \
    data/train data/lang exp/tri5_ali exp/tri6_nnet || exit 1

  touch exp/tri6_nnet/.done
fi
