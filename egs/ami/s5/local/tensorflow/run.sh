#!/bin/bash
mic=ihm
ngram_order=4
model_type=small
stage=1
weight=0.5

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

dir=data/tensorflow/$model_type
mkdir -p $dir

if [ $stage -le 1 ]; then
  local/tensorflow/prep_data.sh $dir
fi

mkdir -p $dir/
if [ $stage -le 2 ]; then
  $decode_cmd $dir/train.log python local/tensorflow/rnnlm.py --data_path=$dir --model=$model_type --save_path=$dir/rnnlm --vocab_path=$dir/wordlist.rnn.final
fi

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/lmrescore_rnnlm_lat.sh \
      --cmd "$tensorflow_cmd --mem 16G" \
      --rnnlm-ver tensorflow  --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}.tfrnnlm.lat.${ngram_order}gram.$weight  &

  done
fi

wait
