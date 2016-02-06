#!/bin/bash


# This runs on the full training set (with duplicates removed), with p-norm
# units, on top of fMLLR features, on GPU.

temp_dir=
dir=nnet2_5
has_fisher=true

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll
                          # likely have to change it.

( 
  if [ ! -f exp/$dir/final.mdl ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d exp/$dir/egs/storage ]; then
      # spread the egs over various machines. 
      utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/exp/$dir/egs/storage exp/$dir/egs/storage
    fi

    steps/nnet2/train_pnorm_accel2.sh --parallel-opts "$parallel_opts" \
      --cmd "$decode_cmd" --stage -10 \
      --num-threads 1 --minibatch-size 512 \
      --mix-up 20000 --samples-per-iter 300000 \
      --num-epochs 15 \
      --initial-effective-lrate 0.005 --final-effective-lrate 0.0002 \
      --num-jobs-initial 3 --num-jobs-final 10 --num-hidden-layers 5 \
      --pnorm-input-dim 5000  --pnorm-output-dim 500 data/train_nodup \
      data/lang exp/tri4_ali_nodup exp/$dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config \
    --transform-dir exp/tri4/decode_eval2000_sw1_tg \
    exp/tri4/graph_sw1_tg data/eval2000 \
    exp/$dir/decode_eval2000_sw1_tg || exit 1;

  if $has_fisher; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      exp/$dir/decode_eval2000_sw1_{tg,fsh_fg} || exit 1;
  fi
)
