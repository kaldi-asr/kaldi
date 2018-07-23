#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li
#           2018  Yiming Wang

# same as lstm_tdnn_1b, but with backstitch training.

# rnnlm/train_rnnlm.sh: best iteration (out of 160) was 156, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 40.2 / 47.8.
# Train objf: -6.47 -5.36 -5.02 -4.85 -4.73 -4.65 -4.59 -4.54 -4.49 -4.45 -4.41 -4.38 -4.35 -4.33 -4.30 -4.29 -4.27 -4.25 -4.23 -4.22 -4.20 -4.19 -4.18 -4.17 -4.16 -4.14 -4.14 -4.13 -4.11 -4.11 -4.10 -4.09 -4.09 -4.07 -4.07 -4.06 -4.05 -4.05 -4.04 -4.04 -4.03 -4.00 -3.98 -3.96 -3.98 -3.96 -3.95 -3.94 -3.95 -3.94 -3.92 -3.92 -3.92 -3.91 -3.90 -3.90 -3.91 -3.90 -3.88 -3.88 -3.89 -3.88 -3.87 -3.86 -3.87 -3.86 -3.85 -3.85 -3.85 -3.85 -3.84 -3.83 -3.84 -3.83 -3.82 -3.82 -3.83 -3.82 -3.81 -3.81 -3.82 -3.81 -3.80 -3.80 -3.80 -3.80 -3.79 -3.79 -3.79 -3.79 -3.78 -3.77 -3.78 -3.77 -3.77 -3.76 -3.77 -3.76 -3.75 -3.75 -3.75 -3.75 -3.75 -3.74 -3.74 -3.74 -3.73 -3.73 -3.73 -3.73 -3.72 -3.73 -3.73 -3.72 -3.71 -3.71 -3.71 -3.71 -3.71 -3.71 -3.72 -3.71 -3.69 -3.70 -3.69 -3.69 -3.69 -3.68 -3.68 -3.68 -3.67 -3.67 -3.67 -3.67 -3.67 -3.66 -3.66 -3.66 -3.65 -3.65 -3.65 -3.65 -3.64 -3.64 -3.64 -3.64 -3.63 -3.63 -3.63 -3.63 -3.63 -3.62 -3.63 -3.62 -3.62 -3.62 -3.62 -3.62 -3.61 -3.61 
# Dev objf:   -11.73 -5.72 -5.18 -4.95 -4.81 -4.72 -4.65 -4.59 -4.55 -4.50 -4.47 -4.44 -4.41 -4.38 -4.36 -4.34 -4.33 -4.31 -4.30 -4.28 -4.26 -4.26 -4.25 -4.23 -4.22 -4.22 -4.22 -4.19 -4.20 -4.18 -4.19 -4.17 -4.16 -4.16 -4.16 -4.14 -4.14 -4.14 -4.13 -4.12 -4.12 -4.07 -4.06 -4.05 -4.04 -4.04 -4.04 -4.03 -4.03 -4.02 -4.02 -4.02 -4.01 -4.01 -4.01 -4.00 -4.00 -4.00 -3.99 -3.99 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.98 -3.98 -3.97 -3.97 -3.97 -3.97 -3.96 -3.96 -3.96 -3.96 -3.96 -3.96 -3.95 -3.95 -3.95 -3.95 -3.95 -3.95 -3.94 -3.94 -3.94 -3.94 -3.94 -3.94 -3.94 -3.94 -3.94 -3.93 -3.93 -3.93 -3.93 -3.93 -3.92 -3.93 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.92 -3.91 -3.91 -3.91 -3.91 -3.91 -3.91 -3.90 -3.90 -3.91 -3.89 -3.89 -3.89 -3.89 -3.89 -3.89 -3.89 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.88 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87 -3.87

# Begin configuration section.
affix=1a
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=40
stage=-10
train_stage=-10
# backstitch options
alpha=0.2 # backstitch training scale
back_interval=1 # backstitch training interval

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd


dir=exp/rnnlm_lstm_tdnn_bs_$affix
text=data/local/dict_nosp_larger/cleaned.gz
wordlist=data/lang_nosp/words.txt
text_dir=data/rnnlm/text_nosp
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 500 lines as dev data.
  gunzip -c $text  | awk -v text_dir=$text_dir '{if(NR%500 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/wsj.txt
fi

if [ $stage -le 1 ]; then
  # the training scripts require that <s>, </s> and <brk> be present in a particular
  # order.
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<SPOKEN_NOISE>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
wsj   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<SPOKEN_NOISE>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=50000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<SPOKEN_NOISE>' \
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
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  backstitch_opt="--backstitch-training-scale $alpha \
    --backstitch-training-interval $back_interval"
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $backstitch_opt $dir
fi

exit 0
