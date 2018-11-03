#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains LMs

#rnnlm/train_rnnlm.sh: best iteration (out of 185) was 181, linking it to final iteration.
#rnnlm/train_rnnlm.sh: train/dev perplexity was 28.6 / 158.5.
#Train objf: -5.10 -4.72 -4.52 -4.40 -4.30 -4.24 -4.17 -4.12 -4.08 -4.04 -4.01 -3.98 -3.95 -3.92 -3.90 -3.88 -3.85 -3.84 -3.83 -3.80 -3.78 -3.77 -3.76 -3.74 -3.72 -3.72 -3.71 -3.70 -3.69 -3.67 -3.58 -3.66 -3.65 -3.64 -3.62 -3.62 -3.62 -3.59 -3.59 -3.58 -3.58 -3.57 -3.57 -3.55 -3.55 -3.54 -3.54 -3.53 -3.51 -3.51 -3.50 -3.49 -3.48 -3.48 -3.48 -3.46 -3.46 -3.46 -3.45 -3.44 -3.42 -3.42 -3.42 -3.41 -3.41 -3.42 -3.42 -3.42 -3.41 -3.41 -3.40 -3.39 -3.38 -3.39 -3.39 -3.30 -3.38 -3.37 -3.37 -3.36 -3.36 -3.36 -3.35 -3.35 -3.36 -3.35 -3.35 -3.35 -3.34 -3.34 -3.33 -3.34 -3.33 -3.32 -3.33 -3.32 -3.32 -3.30 -3.30 -3.30 -3.29 -3.29 -3.31 -3.30 -3.30 -3.29 -3.30 -3.29 -3.29 -3.28 -3.29 -3.28 -3.20 -3.28 -3.27 -3.27 -3.27 -3.26 -3.26 -3.26 -3.26 -3.26 -3.26 -3.26 -3.25 -3.25 -3.24 -3.24 -3.24 -3.24 -3.23 -3.24 -3.23 -3.23 -3.22 -3.22 -3.22 -3.21 -3.22 -3.23 -3.23 -3.22 -3.21 -3.21 -3.21 -3.20 -3.21 -3.20 -3.19 -3.19 -3.19 -3.20 -3.19 -3.19 -3.19 -3.18 -3.19 -3.18 -3.18 -3.17 -3.16 -3.17 -3.16 -3.17 -3.17 -3.17 -3.17 -3.16 -3.15 -3.16 -3.08 -3.16 -3.15 -3.15 -3.15 -3.14 -3.17 -3.16 -3.15 -3.14 -3.14 -3.15 -3.14 -3.14
#Dev objf:   -6.75 -6.30 -6.11 -6.02 -6.01 -5.96 -5.92 -5.87 -5.82 -5.78 -5.77 -5.70 -5.67 -5.71 -5.64 -5.65 -5.63 -5.59 -5.57 -5.56 -5.53 -5.52 -5.50 -5.52 -5.53 -5.50 -5.49 -5.48 -5.47 -5.47 -5.44 -5.43 -5.40 -5.39 -5.39 -5.39 -5.40 -5.41 -5.40 -5.39 -5.39 -5.37 -5.37 -5.36 -5.34 -5.35 -5.34 -5.31 -5.29 -5.29 -5.29 -5.29 -5.27 -5.26 -5.26 -5.26 -5.28 -5.25 -5.25 -5.25 -5.26 -5.26 -5.24 -5.25 -5.23 -5.25 -5.24 -5.22 -5.26 -5.24 -5.21 -5.22 -5.21 -5.22 -5.22 -5.21 -5.21 -5.22 -5.21 -5.20 -5.21 -5.20 -5.21 -5.20 -5.21 -5.20 -5.21 -5.23 -5.19 -5.20 -5.17 -5.17 -5.18 -5.17 -5.17 -5.17 -5.18 -5.17 -5.18 -5.21 -5.19 -5.17 -5.17 -5.16 -5.19 -5.16 -5.17 -5.15 -5.17 -5.15 -5.16 -5.16 -5.14 -5.15 -5.15 -5.14 -5.15 -5.14 -5.14 -5.16 -5.15 -5.17 -5.14 -5.15 -5.14 -5.14 -5.13 -5.13 -5.13 -5.13 -5.12 -5.13 -5.12 -5.13 -5.11 -5.13 -5.15 -5.13 -5.12 -5.11 -5.12 -5.12 -5.11 -5.11 -5.11 -5.11 -5.09 -5.09 -5.09 -5.10 -5.10 -5.10 -5.10 -5.10 -5.10 -5.08 -5.08 -5.10 -5.10 -5.10 -5.08 -5.09 -5.09 -5.08 -5.09 -5.10 -5.09 -5.09 -5.08 -5.09 -5.08 -5.10 -5.08 -5.07 -5.08 -5.08 -5.08 -5.07 -5.09 -5.09 -5.07 -5.08 -5.07 -5.08

# Begin configuration section.

dir=exp/rnnlm_lstm_1e_2
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=0

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false

ac_model_dir=exp/chain/tdnn1c_swbd_sp
decode_dir_suffix=rnnlm_1e_2
ngram_order=6 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

text=data/train/text
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/swbd.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<sil>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
swbd   1   1.0
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
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                  --stage $train_stage --num-epochs 6 --cmd "$train_cmd" $dir
fi

LM=test
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}
    for weight in 0.10 0.20 0.30 0.40; do
      # Lattice rescoring
      rnnlm/lmrescore$pruned.sh \
        --cmd "$decode_cmd" \
        --weight $weight --max-ngram-order $ngram_order \
        data/lang $dir \
        data/${decode_set} ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_${weight}
      done
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd" --N 20 \
      0.8 data/lang $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

# running backward RNNLM, which further improves WERS by combining backward with
# the forward RNNLM trained in this script.
if [ $stage -le 6 ] && $run_backward_rnnlm; then
  local/rnnlm/run_tdnn_lstm_back_1e.sh
fi

exit 0
