#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2018  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 9) was 8, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 94.1 / 155.1.
# Train objf: -6.24 -5.45 -5.12 -4.95 -4.84 -4.74 -4.66 -4.59 -4.52 -4.46
# Dev objf:   -11.92 -5.80 -5.32 -5.17 -5.10 -5.07 -5.05 -5.05 -5.04 -5.06

# 1-pass results 
# %WER 8.3 | 1155 27500 | 92.7 4.9 2.4 1.0 8.3 68.8 | -0.019 | /export/a12/ywang/kaldi/egs/tedlium/s5_r2/exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi/decode_looped_test/score_10_0.0/ctm.filt.filt.sys

# 4-gram rescoring
# %WER 7.8 | 1155 27500 | 93.1 4.5 2.4 0.9 7.8 66.4 | -0.089 | /export/a12/ywang/kaldi/egs/tedlium/s5_r2/exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi/decode_looped_test_rescore/score_10_0.0/ctm.filt.filt.sys

# RNNLM lattice rescoring
# %WER 7.2 | 1155 27500 | 93.6 4.0 2.3 0.8 7.2 64.3 | -0.927 | exp/decode_looped_test_rnnlm_tedlium_rescore//score_10_0.0/ctm.filt.filt.sys

# RNNLM nbest rescoring
# %WER 7.4 | 1155 27500 | 93.4 4.3 2.3 0.9 7.4 64.8 | -0.863 | exp/decode_looped_test_rnnlm_tedlium_nbest_rescore/score_8_0.0/ctm.filt.filt.sys

# Begin configuration section.
cmd=run.pl
decode_cmd=run.pl
dir=exp/rnnlm_lstm_tdnn
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=0
train_stage=-10
epochs=20

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
decode_dir_suffix=rnnlm_tedlium
ac_model_dir=exp/chain_cleaned/tdnn_lstm1i_adversarial1.0_interval4_epoches7_lin_to_5_sp_bi
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

wordlist=data/lang/words.txt
text=data/train/text
dev_sents=10000
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/prepare_data.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $text | cut -d ' ' -f2- | head -n $dev_sents > $text_dir/dev.txt
  cat $text | cut -d ' ' -f2- | tail -n +$[$dev_sents+1] > $text_dir/ted.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
ted   1   1.0
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

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --stage $train_stage --num-epochs $epochs \
                       --cmd "queue.pl" $dir
fi

if [ $stage -le 4 ] && $run_lat_rescore; then
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

if [ $stage -le 5 ] && $run_nbest_rescore; then
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
