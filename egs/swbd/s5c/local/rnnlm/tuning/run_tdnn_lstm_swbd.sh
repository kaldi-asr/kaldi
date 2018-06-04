#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# This script is only for SWBD data (no Fisher).

# rnnlm/train_rnnlm.sh: best iteration (out of 30) was 15, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 44.1 / 59.7.
# Train objf: -5.35 -4.67 -4.40 -4.26 -4.18 -4.11 -4.05 -4.01 -3.97 -3.93 -3.90 -3.87 -3.84 -3.81 -3.79 -3.77 -3.74 -3.73 -3.71 -3.68 -3.66 -3.63 -3.61 -3.59 -3.57 -3.55 -3.53 -3.51 -3.49 -3.47
# Dev objf:   -10.32 -4.96 -4.53 -4.37 -4.30 -4.23 -4.20 -4.16 -4.14 -4.12 -4.15 -4.10 -4.09 -4.09 -4.10 -4.09 -4.10 -4.10 -4.11 -4.12 -4.12 -4.12 -4.13 -4.14 -4.14 -4.17 -4.17 -4.17 -4.18 -4.19

# Begin configuration section.
dir=exp/rnnlm_tdnn_lstm_swbd
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
embedding_l2=0.005 # embedding layer l2 regularize
comp_l2=0.005 # component-level l2 regularize
output_l2=0.005 # output-layer l2 regularize
epochs=30
stage=-10
train_stage=0

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

text=data/train_nodev/text
wordlist=data/lang/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/swbd1_data_prep.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/swbd.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
swbd  1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features 20000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>,[noise],[laughter],[vocalized-noise]' \
                           $dir/config/words.txt > $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-2))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
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
