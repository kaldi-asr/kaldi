#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 3, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 80.6 / 179.5.
# Train objf: -332.70 -5.03 -4.65 -4.39 -4.20 -4.04 -3.91 -3.80 -3.70 -3.62 
# Dev objf:   -10.07 -5.52 -5.25 -5.19 -5.21 -5.24 -5.29 -5.32 -5.35 -5.39 

# Begin configuration section.
dir=exp/rnnlm_lstm_1b
embedding_dim=800
embedding_l2=0.005 # embedding layer l2 regularize
comp_l2=0.005 # component-level l2 regularize
output_l2=0.005 # output-layer l2 regularize
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

lstm_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
lstm-layer name=lstm1 cell-dim=$embedding_dim $lstm_opts
lstm-layer name=lstm2 cell-dim=$embedding_dim $lstm_opts
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
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
  rnnlm/train_rnnlm.sh --embedding_l2 $embedding_l2 \
                       --stage $train_stage \
                       --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
