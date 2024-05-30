#!/usr/bin/env bash

set -x

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li
# Copyright 2019  IBM Corp. (author: Michael Picheny) modified for MALACH data

# rnnlm/train_rnnlm.sh: train/dev perplexity was 50.6 / 96.4.
# Train objf: -4.59 -4.40 -4.30 -4.23 -4.18 -4.14 -4.11 -4.08 -4.05 -4.03 -4.00 -3.99 -3.98 -3.96 -3.94 -3.93 -3.92 -3.91 -3.90 
# Dev objf:   -5.12 -4.84 -4.75 -4.71 -4.64 -4.61 -4.60 -4.58 -4.60 -4.57 -4.59 -4.59 -4.58 -4.57 -4.57 -4.57 -4.57 -4.57 -4.57 

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1a
embedding_dim=200
epochs=60
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

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
lstm-layer name=lstm1 cell-dim=$embedding_dim 
relu-renorm-layer name=tdnn dim=$embedding_dim input=Append(0, IfDefined(-1))
lstm-layer name=lstm2 cell-dim=$embedding_dim 
output-layer name=output include-log-softmax=false dim=$embedding_dim
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
  rnnlm/train_rnnlm.sh --stage $train_stage \
                       --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
