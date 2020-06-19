#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# 2-layer LSTM hidden 200:
# rnnlm/train_rnnlm.sh: best iteration (out of 18) was 10, linking it to final iteration.
# Train objf: -620.20 -4.49 -4.33 -4.21 -4.13 -4.06 -4.00 -3.96 -3.92 -3.88 -3.84 -3.81 -3.77 -3.74 -3.71 -3.68 -3.65 -3.59
# Dev objf:   -10.76 -4.74 -4.54 -4.44 -4.38 -4.33 -4.31 -4.29 -4.28 -4.28 -4.28 -4.29 -4.29 -4.30 -4.32 -4.33 -4.34 -4.36

# 2-layer LSTM + 1 TDNN layer hidden 200:
# rnnlm/train_rnnlm.sh: best iteration (out of 18) was 7, linking it to final iteration.  
# Train objf: -664.80 -4.39 -4.21 -4.09 -4.01 -3.95 -3.89 -3.85 -3.81 -3.77 -3.74 -3.71 -3.68 -3.65 -3.63 -3.61 -3.59 -3.57
# Dev objf:   -10.76 -4.65 -4.45 -4.34 -4.30 -4.29 -4.28 -4.27 -4.27 -4.28 -4.28 -4.29 -4.29 -4.30 -4.31 -4.32 -4.33 -4.34

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1a
embedding_dim=200
epochs=60
mic=sdm1
stage=-10
train_stage=0

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

train=data/$mic/train/text
dev=data/$mic/dev/text
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
