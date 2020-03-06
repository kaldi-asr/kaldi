#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2018  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 60) was 58, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 25.1 / 104.5.
# Train objf: -3.60 -3.52 -3.48 -3.44 -3.41 -3.38 -3.36 -3.35 -3.33 -3.31 -3.29 -3.29 -3.28 -3.28 -3.27 -3.25 -3.25 -3.23 -3.23 -3.22 -3.22 -3.21 -3.20 -3.19 -3.19 -3.18 -3.18 -3.18 -3.17 -3.17 -3.15 -3.15 -3.15 -3.15 -3.14 -3.14 -3.12 -3.14 -3.16 -3.13 -3.12 -3.13 -3.11 -3.12 -3.11 -3.10 -3.09 -3.10 -3.06 -3.08 -3.10 -3.09 -3.08 -3.09 -3.02 -3.01 -3.02 -2.98 -3.02
# Dev objf:   -5.12 -5.04 -4.98 -4.93 -4.91 -4.89 -4.87 -4.86 -4.82 -4.80 -4.79 -4.79 -4.78 -4.77 -4.78 -4.76 -4.76 -4.75 -4.75 -4.74 -4.74 -4.73 -4.73 -4.72 -4.71 -4.72 -4.71 -4.70 -4.70 -4.70 -4.70 -4.70 -4.70 -4.70 -4.69 -4.69 -4.68 -4.68 -4.67 -4.67 -4.68 -4.67 -4.67 -4.67 -4.67 -4.67 -4.67 -4.67 -4.66 -4.68 -4.68 -4.72 -4.68 -4.66 -4.71 -4.65 -4.65 -4.65 -4.65

# 1-pass results 
# %WER 8.3 | 1155 27500 | 92.7 4.9 2.4 1.0 8.3 68.8 | -0.019 | /export/a12/ywang/kaldi/egs/tedlium/s5_r2/exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi/decode_looped_test/score_10_0.0/ctm.filt.filt.sys

# 4-gram rescoring
# %WER 7.8 | 1155 27500 | 93.1 4.5 2.4 0.9 7.8 66.4 | -0.089 | /export/a12/ywang/kaldi/egs/tedlium/s5_r2/exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi/decode_looped_test_rescore/score_10_0.0/ctm.filt.filt.sys

# RNNLM lattice rescoring
# %WER 6.8 | 1155 27500 | 94.0 3.7 2.3 0.8 6.8 62.3 | -0.844 | exp/decode_looped_test_rnnlm_lm1b_tedlium_weight3_rescore//score_10_0.0/ctm.filt.filt.sys

# RNNLM nbest rescoring
# %WER 6.9 | 1155 27500 | 94.0 3.8 2.2 0.9 6.9 61.3 | -0.769 | exp/decode_looped_test_rnnlm_lm1b_tedlium_weight3_nbest_rescore//score_10_0.0/ctm.filt.filt.sys

# Begin configuration section.
cmd=run.pl
decode_cmd=run.pl
dir=exp/rnnlm_lstm_tdnn_with_lm1b
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=0
train_stage=-10
epochs=20

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
decode_dir_suffix=rnnlm_lstm_tdnn_with_lm1b
ac_model_dir=exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

lm1b_dir=data/rnnlm/lm1b
wordlist=data/lang/words.txt
train_text=data/train/text
dev_sents=10000
text_dir=data/rnnlm/text_lm1b_tedlium
mkdir -p $dir/config
set -e

for f in $wordlist $train_text; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; generate lm1b data first; \
    search for local/prepare_data.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
    mkdir -p $lm1b_dir
    cd $lm1b_dir
    if [ ! -f training-monolingual.tgz ]; then
        wget http://statmt.org/wmt11/training-monolingual.tgz .
    fi
    echo "Downloaded google one billion dataset."
    
    if [ ! -d training-monolingual ]; then
        tar --extract -v --file training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
    fi
    echo "Untar google one billion dataset."

    for year in 2007 2008 2009 2010 2011; do 
        cat training-monolingual/news.${year}.en.shuffled
    done | sort -u --output=training-monolingual/news.20XX.en.shuffled.sorted
    echo "Done sorting corpus."

    time cat training-monolingual/news.20XX.en.shuffled.sorted | \
    ../../../utils/normalize_punctuation.pl -l en -q 1 | \
    ../../../utils/tokenizer.pl -l en -q 1 > \
    training-monolingual/news.20XX.en.shuffled.sorted.tokenized
    echo "Done tokenizing corpus."
    cd ../../..
fi

if [ $stage -le 1 ]; then
  mkdir -p $text_dir
  cat $train_text | cut -d ' ' -f2- | head -n $dev_sents > $text_dir/dev.txt
  cat $train_text | cut -d ' ' -f2- | tail -n +$[$dev_sents+1] > $text_dir/ted.txt
  cp $lm1b_dir/training-monolingual/news.20XX.en.shuffled.sorted.tokenized $text_dir/lm1b.txt
fi

if [ $stage -le 2 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
ted   1   3.0
lm1b    1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=10000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>' \
                           $dir/config/words.txt > $dir/config/features.txt
fi

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

if [ $stage -le 3 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             --words_per_split 100000000 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 4 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 5 \
                       --stage $train_stage \
                       --num-epochs $epochs --cmd "queue.pl" $dir
fi

if [ $stage -le 5 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in dev test; do
    decode_dir=${ac_model_dir}/decode_looped_${decode_set}_rescore

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang $dir \
      data/${decode_set}_hires ${decode_dir} \
      exp/decode_looped_${decode_set}_${decode_dir_suffix}_rescore
  done
fi

if [ $stage -le 6 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in dev test; do
    decode_dir=${ac_model_dir}/decode_looped_${decode_set}_rescore

    # nbest rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      0.8 data/lang $dir \
      data/${decode_set}_hires ${decode_dir} \
      exp/decode_looped_${decode_set}_${decode_dir_suffix}_nbest_rescore
  done
fi

exit 0
