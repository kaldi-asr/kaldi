#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2018  Ke Li

# This script trains an RNNLM on the SEAME training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 13) was 5, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 45.8 / 69.1.
# Train objf: -4.45 -4.20 -4.04 -3.90 -3.78 -3.66 -3.54 -3.42 -3.30 -3.19 -3.08 -2.98
# Dev objf:   -4.74 -4.42 -4.32 -4.25 -4.24 -4.25 -4.29 -4.32 -4.38 -4.43 -4.50 -4.56

# WER summary on dev and test sets
# System                   tdnn_1c_sp  +lattice_rescore  +nbest_rescore 
# WER on dev_man             19.96          18.75             18.80
# WER on dev_sge             25.72          24.26             24.28


# %WER 19.96 [ 16895 / 84652, 1656 ins, 3649 del, 11590 sub ] exp/chain/tdnn_1a_sp/decode_dev_man/wer_10_0.0
# %WER 25.72 [ 13992 / 54408, 1459 ins, 2939 del, 9594 sub ] exp/chain/tdnn_1a_sp/decode_dev_sge/wer_10_0.0

# Lattice rescoring
# %WER 18.75 [ 15872 / 84652, 1688 ins, 3450 del, 10734 sub ] exp/chain/tdnn_1a_sp/decode_dev_man_rnnlm_1a_rescore/wer_10_0.0
# %WER 24.26 [ 13198 / 54408, 1482 ins, 2782 del, 8934 sub ] exp/chain/tdnn_1a_sp/decode_dev_sge_rnnlm_1a_rescore/wer_10_0.0

# Nbest rescoring
# %WER 18.80 [ 15917 / 84652, 1541 ins, 3671 del, 10705 sub ] exp/chain/tdnn_1a_sp/decode_dev_man_rnnlm_1a_nbest_rescore/wer_10_0.0
# %WER 24.28 [ 13211 / 54408, 1406 ins, 2881 del, 8924 sub ] exp/chain/tdnn_1a_sp/decode_dev_sge_rnnlm_1a_nbest_rescore/wer_10_0.0

# Begin configuration section.

dir=exp/rnnlm_lstm_1a
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10
epochs=40

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false
ac_model_dir=exp/chain/tdnn_1a_sp
decode_dir_suffix=rnnlm_1a
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

# test of 960 hours training transcriptions
text=data/train/text
lexicon=data/lang_nosp/words.txt
text_dir=data/rnnlm/text_rnnlm_1a
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/seame.txt
fi

if [ $stage -le 1 ]; then
  cp $lexicon $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
seame   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --top-word-features=3000 \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<unk>,<noise>,<v-noise>' \
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
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200 \
                            $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-final 1 \
                       --stage $train_stage \
                       --num-epochs $epochs \
                       --cmd "$train_cmd" $dir
fi

if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
#  LM=tgsmall # if using the original 3-gram G.fst as old lm
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in dev_man dev_sge; do
    decode_dir=${ac_model_dir}/decode_${decode_set}
    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
        --cmd "$decode_cmd --mem 8G" \
        --weight 0.45 --max-ngram-order $ngram_order \
        data/lang_test $dir \
        data/${decode_set}_hires ${decode_dir} \
        ${ac_model_dir}/decode_${decode_set}_${decode_dir_suffix}_rescore
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in dev_man dev_sge; do
    decode_dir=${ac_model_dir}/decode_${decode_set}
    # Nbest rescoring
    rnnlm/lmrescore_nbest.sh \
        --cmd "$decode_cmd --mem 8G" --N 20 \
        0.4 data/lang_test $dir \
        data/${decode_set}_hires ${decode_dir} \
        ${ac_model_dir}/decode_${decode_set}_${decode_dir_suffix}_nbest_rescore
  done
fi

exit 0;
