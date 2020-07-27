#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li
# Copyright 2019  IBM Corp. (author: Michael Picheny) modified for MALACH data


# This script is similar to rnnlm_lstm_tdnn_a.sh except for adding L2 regularization.

# rnnlm/train_rnnlm.sh: best iteration (out of 30) was 28, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 54.2 / 90.7.
# Train objf: -4.59 -4.41 -4.31 -4.25 -4.21 -4.18 -4.15 -4.13 -4.12 -4.10 -4.09 -4.08 -4.07 -4.06 -4.05 -4.05 -4.04 -4.04 -4.03 -4.02 -4.02 -4.01 -4.00 -4.00 -3.99 -3.99 -3.98 -3.98 -3.98 
# Dev objf:   -5.16 -4.92 -4.78 -4.76 -4.64 -4.63 -4.61 -4.59 -4.57 -4.57 -4.57 -4.55 -4.55 -4.57 -4.56 -4.54 -4.58 -4.53 -4.55 -4.55 -4.54 -4.51 -4.53 -4.52 -4.52 -4.52 -4.52 -4.51 -4.53 

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1b
embedding_dim=200
embedding_l2=0.005 # embedding layer l2 regularize
comp_l2=0.005 # component-level l2 regularize
output_l2=0.005 # output-layer l2 regularize
epochs=90
stage=-10
train_stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

train=data/train/text
dev=data/dev/text
wordlist=data/lang/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $train $dev $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train | cut -d ' ' -f2- > $text_dir/malach.txt
  cat $dev | cut -d ' ' -f2- > $text_dir/dev.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
malach  1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features 10000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>,[noise],[laughter]' \
                           $dir/config/words.txt > $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
lstm-layer name=lstm1 cell-dim=$embedding_dim $lstm_opts
relu-renorm-layer name=tdnn dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
lstm-layer name=lstm2 cell-dim=$embedding_dim $lstm_opts
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --embedding_l2 $embedding_l2 \
                       --stage $train_stage \
                       --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
