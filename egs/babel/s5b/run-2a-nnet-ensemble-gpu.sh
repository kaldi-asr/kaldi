#!/bin/bash

. ./lang.conf
. conf/common_vars.sh

train_stage=-10
dir=exp/tri6b_nnet

. ./utils/parse_options.sh

set -e
set -o pipefail
set -u

dnn_num_hidden_layers=4
dnn_pnorm_input_dim=3000
dnn_pnorm_output_dim=300
dnn_init_learning_rate=0.004
dnn_final_learning_rate=0.001
temp_dir=`pwd`/nnet_gpu_egs
ensemble_size=4
initial_beta=0.1
final_beta=5
egs_dir=

# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.
echo "Waiting till exp/tri5_ali/.done exists...."
while [ ! -f exp/tri5_ali/.done ]; do sleep 30; done
echo "...done waiting for exp/tri5_ali/.done"

if [ ! -f $dir/.done ]; then
  steps/nnet2/train_pnorm_ensemble.sh \
    --stage $train_stage --mix-up $dnn_mixup --egs-dir "$egs_dir" \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_pnorm_input_dim \
    --pnorm-output-dim $dnn_pnorm_output_dim \
    --cmd "$train_cmd" \
    "${dnn_gpu_parallel_opts[@]}" \
    --ensemble-size $ensemble_size --initial-beta $initial_beta --final-beta $final_beta \
    data/train data/lang exp/tri5_ali $dir || exit 1
  touch $dir/.done
fi

