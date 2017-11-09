#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# This script is similar to rnnlm_lstm_tdnn_a except for adding L2 regularization.

# rnnlm/train_rnnlm.sh: best iteration (out of 50) was 46, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 38.6 / 45.8. 
# Train objf: -472.10 -4.26 -4.13 -4.06 -4.01 -3.98 -3.94 -3.92 -3.90 -3.88 -3.87 -3.85 -3.83 -3.82 -3.81 -3.81 -3.79 -3.78 -3.78 -3.77 -3.77 -3.75 -3.75 -3.75 -3.74 -3.74 -3.73 -3.72 -3.72 -3.71 -3.71 -3.70 -3.70 -3.70 -3.69 -3.69 -3.69 -3.68 -3.68 -3.68 -3.68 -3.67 -3.67 -3.66 -3.66 -3.66 -3.65 -3.65 -3.65 -3.65 
# Dev objf:   -10.65 -4.63 -4.38 -4.26 -4.20 -4.16 -4.12 -4.09 -4.07 -4.04 -4.03 -3.99 -3.98 -3.99 -3.96 -3.94 -3.94 -3.94 -3.92 -3.91 -3.91 -3.90 -3.89 -3.89 -3.88 -3.88 -3.87 -3.87 -3.86 -3.86 -3.85 -3.85 -3.85 -3.85 -3.85 -3.84 -3.84 -3.85 -3.84 -3.84 -3.83 -3.83 -3.83 -3.83 -3.83 -3.83 -3.82 -3.83 -3.83 -3.83 

# Begin configuration section.
cmd=run.pl
dir=exp/rnnlm_lstm_tdnn_b
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=10
stage=-10
train_stage=0

. utils/parse_options.sh

text=data/train/text
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
  # hold out one in every 500 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%500 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/swbd.txt
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
                           --special-words='<s>,</s>,<brk>,<unk>,[noise],[laughter]' \
                           $dir/config/words.txt > $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-batchnorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-batchnorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-2))
relu-batchnorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
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
                       --num-epochs $epochs --cmd "queue.pl" $dir
fi

exit 0
