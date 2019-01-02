#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# This script is similar to rnnlm_lstm_tdnn_a.sh except for adding L2 regularization.

# local/rnnlm/train_rnnlm.sh: best iteration (out of 18) was 17, linking it to final iteration.
# local/rnnlm/train_rnnlm.sh: train/dev perplexity was 45.6 / 68.7.
# Train objf: -651.50 -4.44 -4.26 -4.15 -4.08 -4.03 -4.00 -3.97 -3.94 -3.92 -3.90 -3.89 -3.88 -3.86 -3.85 -3.84 -3.83 -3.82
# Dev objf:   -10.76 -4.68 -4.47 -4.38 -4.33 -4.29 -4.28 -4.27 -4.26 -4.26 -4.25 -4.24 -4.24 -4.24 -4.23 -4.23 -4.23 -4.23

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1b
embedding_dim=200
embedding_l2=0.005 # embedding layer l2 regularize
comp_l2=0.005 # component-level l2 regularize
output_l2=0.005 # output-layer l2 regularize
epochs=90
mic=
stage=-10
train_stage=0

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

train=data/train/text
dev=data/dev2/text   # We at no stage in run.sh should decode dev2 partition for results!
wordlist=data/lang/words.txt
text_dir=data/local/rnnlm/text
mkdir -p $dir/config
set -e

for f in $train $dev $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train | cut -d ' ' -f2- > $text_dir/ami.txt
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
ami  1   1.0
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
