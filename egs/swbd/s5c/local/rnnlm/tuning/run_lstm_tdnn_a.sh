#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 40) was 39, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 40.8 / 46.2.
# Train objf: -313.40 -4.42 -4.24 -4.16 -4.10 -4.06 -4.02 -3.99 -3.96 -3.95 -3.93 -3.91 -3.89 -3.88 -3.87 -3.86 -3.85 -3.84 -3.83 -3.82 -3.82 -3.80 -3.80 -3.79 -3.79 -3.78 -3.77 -3.77 -3.76 -3.76 -3.75 -3.74 -3.74 -3.74 -3.73 -3.73 -3.72 -3.72 -3.71 -3.71
# Dev objf:   -10.65 -4.66 -4.36 -4.25 -4.17 -4.13 -4.09 -4.05 -4.04 -4.02 -4.00 -3.99 -3.97 -3.96 -3.95 -3.93 -3.93 -3.92 -3.91 -3.91 -3.90 -3.90 -3.89 -3.88 -3.88 -3.87 -3.87 -3.87 -3.86 -3.86 -3.85 -3.85 -3.85 -3.85 -3.85 -3.84 -3.84 -3.84 -3.84 -3.83

# Begin configuration section.
cmd=run.pl
dir=exp/rnnlm_lstm_tdnn_a
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
epochs=8
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

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-2))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-1))
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
                       --num-epochs $epochs --cmd "queue.pl" $dir
fi

exit 0
