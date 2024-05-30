#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 3, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 69.4 / 183.1.
# Train objf: -333.60 -4.98 -4.54 -4.24 -3.98 -3.76 -3.56 -3.39 -3.25 -3.13
# Dev objf:   -10.07 -5.53 -5.23 -5.21 -5.27 -5.37 -5.47 -5.57 -5.68 -5.77

# Begin configuration section.
dir=exp/rnnlm_lstm_1a
embedding_dim=800
epochs=160
stage=-10
train_stage=-10

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd


text=data/train/text
wordlist=data/lang/words.txt
dev_sents=3000
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/prepare_data.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $text | cut -d ' ' -f2- | head -n $dev_sents> $text_dir/dev.txt
  cat $text | cut -d ' ' -f2- | tail -n +$[$dev_sents+1] > $text_dir/hub.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l` 
  echo "<brk> $n" >> $dir/config/words.txt 

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
hub   1   1.0
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
                           --special-words='<s>,</s>,<brk>,<unk>' \
                           $dir/config/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
lstm-layer name=lstm1 cell-dim=$embedding_dim
lstm-layer name=lstm2 cell-dim=$embedding_dim
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 100.0 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --stage $train_stage \
                       --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
