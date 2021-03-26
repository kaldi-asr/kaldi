#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2018  Ke Li

# This script trains LMs on the librispeech-lm-norm.txt.gz.

# rnnlm/train_rnnlm.sh: best iteration (out of 143) was 142, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 109.2 / 110.7.
# Train objf: -5.74 -5.54 -5.44 -5.37 -5.32 -5.28 -5.25 -5.23 -5.20 -5.18 -5.15 -5.14 -5.12 -5.10 -5.09 -5.08 -5.07 -5.05 -5.04 -5.04 -5.03 -5.02 -5.01 -5.00 -4.99 -4.99 -4.98 -4.97 -4.96 -4.96 -4.95 -4.95 -4.94 -4.93 -4.93 -4.92 -4.92 -4.92 -4.91 -4.90 -4.90 -4.89 -4.89 -4.89 -4.88 -4.88 -4.87 -4.87 -4.87 -4.86 -4.86 -4.86 -4.85 -4.85 -4.84 -4.84 -4.84 -4.84 -4.84 -4.83 -4.83 -4.83 -4.82 -4.82 -4.82 -4.82 -4.81 -4.81 -4.81 -4.81 -4.80 -4.80 -4.80 -4.79 -4.79 -4.79 -4.79 -4.78 -4.79 -4.78 -4.78 -4.78 -4.78 -4.77 -4.77 -4.77 -4.77 -4.77 -4.76 -4.76 -4.76 -4.76 -4.76 -4.75 -4.75 -4.75 -4.75 -4.75 -4.74 -4.74 -4.74 -4.74 -4.74 -4.74 -4.73 -4.74 -4.74 -4.73 -4.73 -4.73 -4.73 -4.73 -4.72 -4.73 -4.73 -4.73 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.71 -4.71 -4.71 -4.71 -4.71 -4.70 -4.70 -4.70 -4.70 -4.70 -4.69 -4.69 -4.69 -4.69 -4.69 -4.69 -4.68 -4.68
# Dev objf:   -5.99 -5.65 -5.53 -5.44 -5.38 -5.34 -5.30 -5.27 -5.22 -5.20 -5.18 -5.16 -5.14 -5.12 -5.11 -5.10 -5.09 -5.08 -5.07 -5.05 -5.04 -5.04 -5.03 -5.01 -5.00 -4.99 -4.99 -4.98 -4.97 -4.97 0.00 -4.96 -4.95 -4.95 -4.94 -4.93 -4.93 -4.92 -4.92 -4.91 -4.91 -4.90 -4.90 -4.89 -4.89 -4.89 -4.88 -4.88 -4.88 -4.87 -4.87 -4.87 -4.86 -4.86 -4.85 -4.85 -4.87 -4.84 -4.84 -4.84 -4.83 -4.91 -4.83 -4.83 -4.83 -4.82 -4.82 -4.82 -4.82 -4.81 -4.81 -4.81 -4.80 -4.80 -4.80 -4.80 -4.80 -4.79 -4.79 -4.79 -4.79 -4.79 -4.79 -4.78 -4.78 -4.79 -4.78 -4.77 -4.77 -4.77 -4.77 -4.77 -4.77 -4.77 -4.76 -4.76 -4.76 -4.76 -4.76 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.75 -4.74 -4.74 -4.74 -4.74 -4.74 -4.74 -4.74 -4.73 -4.74 -4.73 -4.73 -4.73 -4.73 -4.73 -4.73 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.72 -4.71 -4.71 -4.71 -4.71 -4.71 -4.71 -4.71 -4.71

# WER summary on dev and test sets
# System                      tdnn_1d_sp  +lattice_rescore  +nbest_rescore 
# WER on dev(fglarge)              3.34         2.71            2.62
# WER on dev(tglarge)              3.44         2.75            2.66
# WER on dev_other(fglarge)        8.70         7.37            7.55
# WER on dev_other(tglarge)        9.25         7.56            7.73
# WER on test(fglarge)             3.77         3.12            3.06
# WER on test(tglarge)             3.85         3.18            3.11
# WER on test_other(fglarge)       8.91         7.63            7.68
# WER on test_other(tglarge)       9.31         7.83            7.95

# command to get the WERs above:
# tdnn_1d_sp
# for test in dev_clean test_clean dev_other test_other; do for lm in fglarge tglarge; do grep WER exp/chain_cleaned/tdnn_1d_sp/decode_${test}_${lm}/wer* | best_wer.sh; done; done
# tdnn_1d_sp with lattice rescoring 
# for test in dev_clean test_clean dev_other test_other; do for lm in fglarge tglarge; do grep WER exp/chain_cleaned/tdnn_1d_sp/decode_${test}_${lm}_rnnlm_1a_rescore/wer* | best_wer.sh; done; done
# tdnn_1d_sp with nbest rescoring 
# for test in dev_clean test_clean dev_other test_other; do for lm in fglarge tglarge; do grep WER exp/chain_cleaned/tdnn_1d_sp/decode_${test}_${lm}_rnnlm_1a_nbest_rescore/wer* | best_wer.sh; done; done

# Begin configuration section.

dir=exp/rnnlm_lstm_1a
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10
epochs=4

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false
ac_model_dir=exp/chain_cleaned/tdnn_1d_sp
decode_dir_suffix=rnnlm_1a
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

text=data/local/lm/librispeech-lm-norm.txt.gz
lexicon=data/lang_nosp/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  if [ ! -f $text ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm 
  fi
  echo -n >$text_dir/dev.txt
  # hold out one in every 2000 lines as dev data.
  gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/librispeech.txt
fi

if [ $stage -le 1 ]; then
  cp $lexicon $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
librispeech   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --top-word-features=5000 \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<UNK>,<SPOKEN_NOISE>' \
                           $dir/config/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 400 \
                            $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-final 8 \
                       --stage $train_stage \
                       --num-epochs $epochs \
                       --cmd "$train_cmd" $dir
fi

exit 0
