#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains a backward LMs on the swbd LM-training data, and use it
# to rescore either decoded lattices, or lattices that are just rescored with
# a forward RNNLM. In order to run this, you must first run the forward RNNLM
# recipe at local/rnnlm/run_tdnn_lstm.sh

#rnnlm/train_rnnlm.sh: best iteration (out of 185) was 179, linking it to final iteration.
#rnnlm/train_rnnlm.sh: train/dev perplexity was 28.7 / 124.1.
#Train objf: -5.09 -4.73 -4.53 -4.40 -4.31 -4.24 -4.17 -4.12 -4.07 -4.04 -4.01 -3.98 -3.95 -3.93 -3.90 -3.88 -3.86 -3.84 -3.83 -3.81 -3.79 -3.77 -3.76 -3.74 -3.73 -3.72 -3.71 -3.70 -3.69 -3.68 -3.58 -3.66 -3.66 -3.65 -3.62 -3.62 -3.62 -3.60 -3.60 -3.58 -3.58 -3.58 -3.57 -3.55 -3.55 -3.54 -3.54 -3.53 -3.52 -3.51 -3.50 -3.50 -3.49 -3.48 -3.48 -3.47 -3.46 -3.46 -3.45 -3.44 -3.43 -3.43 -3.42 -3.41 -3.41 -3.43 -3.42 -3.42 -3.41 -3.41 -3.40 -3.39 -3.39 -3.39 -3.39 -3.30 -3.38 -3.37 -3.37 -3.36 -3.36 -3.36 -3.35 -3.36 -3.36 -3.35 -3.35 -3.35 -3.34 -3.34 -3.33 -3.34 -3.33 -3.33 -3.33 -3.32 -3.32 -3.31 -3.31 -3.31 -3.30 -3.30 -3.31 -3.31 -3.30 -3.30 -3.30 -3.29 -3.29 -3.28 -3.29 -3.28 -3.20 -3.28 -3.27 -3.27 -3.26 -3.26 -3.26 -3.25 -3.26 -3.26 -3.26 -3.26 -3.25 -3.25 -3.25 -3.24 -3.25 -3.24 -3.23 -3.24 -3.23 -3.23 -3.22 -3.22 -3.22 -3.21 -3.22 -3.23 -3.23 -3.22 -3.21 -3.21 -3.21 -3.20 -3.21 -3.20 -3.19 -3.19 -3.19 -3.20 -3.19 -3.19 -3.19 -3.18 -3.19 -3.18 -3.18 -3.17 -3.16 -3.17 -3.16 -3.17 -3.17 -3.17 -3.17 -3.16 -3.15 -3.16 -3.08 -3.16 -3.16 -3.15 -3.15 -3.14 -3.16 -3.15 -3.15 -3.15 -3.14 -3.15 -3.14 -3.15
#Dev objf:   -6.50 -6.35 -6.04 -5.93 -5.84 -5.76 -6.12 -5.69 -5.70 -5.64 -5.57 -5.57 -5.57 -8.64 -5.51 -6.09 -5.50 -5.78 -5.43 -5.43 -5.44 -5.38 -5.58 -5.39 -6.27 -5.33 -5.57 -5.35 -5.33 -5.46 -5.36 -5.33 -5.62 -5.28 -5.30 -5.28 -5.29 -5.27 -5.24 -5.25 -5.28 -5.26 -5.23 -5.31 -6.08 -5.22 -5.21 -5.16 -5.15 -5.16 -5.14 -5.15 -5.16 -5.15 -5.13 -5.12 -5.12 -5.12 -5.12 -5.10 -5.12 -5.13 -5.12 -5.12 -5.13 -5.08 -5.11 -5.11 -5.09 -5.10 -5.08 -5.08 -5.11 -5.08 -5.06 -5.07 -5.07 -5.06 -5.06 -5.05 -5.09 -5.08 -5.05 -5.04 -5.04 -5.05 -5.03 -5.05 -5.06 -5.04 -5.02 -5.03 -5.01 -5.01 -5.01 -5.01 -5.00 -5.02 -5.03 -5.02 -5.03 -5.00 -5.02 -5.01 -5.01 -4.99 -4.99 -4.98 -4.99 -4.96 -4.96 -4.97 -4.99 -4.98 -4.97 -4.95 -4.95 -4.94 -5.09 -4.95 -4.95 -4.95 -4.96 -4.93 -4.96 -4.95 -4.92 -4.92 -4.94 -4.91 -4.93 -4.92 -4.94 -4.91 -4.93 -4.90 -4.94 -4.92 -4.91 -4.90 -4.92 -4.91 -4.89 -4.90 -4.88 -4.88 -4.89 -4.88 -4.88 -4.88 -4.89 -4.88 -4.88 -4.88 -4.89 -4.87 -4.85 -4.86 -4.86 -4.86 -4.86 -4.87 -4.86 -4.85 -4.87 -4.86 -4.86 -4.85 -4.85 -4.85 -4.86 -4.85 -4.85 -4.84 -4.86 -4.83 -4.83 -4.83 -4.82 -4.82 -4.83 -4.83 -4.83 -4.84

# Begin configuration section.

dir=exp_yomdle_chinese/rnnlm_lstm_1e_backward
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
ac_model_dir=exp_yomdle_chinese/chain/cnn_e2eali_1b
decode_dir_suffix_forward=rnnlm_1e3
decode_dir_suffix_backward=rnnlm_1e3_back
ngram_order=6 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from
              # exploding exponentially

. ./cmd.sh
. ./utils/parse_options.sh

text=data_yomdle_chinese/train/text
dev_text=data_yomdle_chinese/local/local_lm_lr/data/text/dev.txt
extra_text=data_yomdle_chinese/local/local_lm_lr/data/text/extra_lm.txt
lexicon=data_yomdle_chinese/local/dict/lexiconp.txt
text_dir=data_yomdle_chinese/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cp $text $text_dir/train.txt
  cp $dev_text $text_dir/dev.txt
  cp $extra_text $text_dir/extra_lm.txt
fi

if [ $stage -le 1 ]; then
  cp data_yomdle_chinese/lang_test/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<sil>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   3   1.0
extra_lm   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<sil>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<sil>,<brk>' \
                           $dir/config/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                  --stage $train_stage --num-epochs 5 --cmd "$gpu_cmd" $dir
fi

LM=test
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"

  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_lr
    if [ ! -d ${decode_dir}_${decode_dir_suffix_forward}_0.45 ]; then
      echo "$0: Must run the forward recipe first at local/rnnlm/run_tdnn_lstm.sh"
      exit 1
    fi

    for weight in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
      # Lattice rescoring
      rnnlm/lmrescore_back.sh \
        --cmd "$cmd --mem 4G" \
        --weight $weight --max-ngram-order $ngram_order \
        data_yomdle_chinese/lang_$LM $dir \
        data_yomdle_chinese/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_0.45 \
        ${decode_dir}_${decode_dir_suffix_backward}_${weight}
    done
  done
fi

exit 0
