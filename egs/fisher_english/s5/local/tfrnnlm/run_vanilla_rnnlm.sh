#!/bin/bash
ngram_order=4 # this option when used, the rescoring binary makes an approximation
    # to merge the states of the FST generated from RNNLM. e.g. if ngram-order = 4
    # then any history that shares last 3 words would be merged into one state
stage=1
weight=0.5   # when we do lattice-rescoring, instead of replacing the lm-weights
    # in the lattice with RNNLM weights, we usually do a linear combination of
    # the 2 and the $weight variable indicates the weight for the RNNLM scores

train_text=data/train/text
nwords=9999
opts=
dir=data/vanilla_tensorflow

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

mkdir -p $dir

#steps/tfrnnlm/check_tensorflow_installed.sh

if [ $stage -le 1 ]; then
  local/tfrnnlm/rnnlm_data_prep.sh --train-text $train_text --nwords $nwords $dir
fi

mkdir -p $dir
if [ $stage -le 2 ]; then
# the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
  $train_cmd --gpu 1 --mem 20G $dir/train_rnnlm.log utils/parallel/limit_num_gpus.sh \
    python steps/tfrnnlm/vanilla_rnnlm.py --data-path=$dir --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final ${opts}
fi

exit 0

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/lmrescore_rnnlm_lat.sh \
      --cmd "$tfrnnlm_cmd --mem 16G" \
      --rnnlm-ver tensorflow  --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}.vanilla.tfrnnlm.lat.${ngram_order}gram.$weight  &

  done
fi

wait
