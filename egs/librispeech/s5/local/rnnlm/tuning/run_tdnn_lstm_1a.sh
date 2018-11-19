#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2018  Ke Li

# This script trains LMs on the librispeech 960 hours training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 26) was 21, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 118.4 / 152.6.
# Train objf: -5.74 -5.51 -5.38 -5.29 -5.22 -5.16 -5.12 -5.08 -5.05 -5.02 -4.99 -4.97 -4.97 -4.93 -4.90 -4.87 -4.84 -4.82 -4.79 -4.77 -4.75 -4.73 -4.71 -4.69 -4.67
# Dev objf:   -6.00 -5.61 -5.45 -5.36 -5.29 -5.24 -5.20 -5.18 -5.16 -5.13 -5.12 -5.11 -5.11 -5.09 -5.07 -5.06 -5.05 -5.04 -5.03 -5.03 -5.03 -5.03 -5.03 -5.03 -5.03 -5.03

# WER summary on dev and test sets
# System                      tdnn_1d_sp  +lattice_rescore  +nbest_rescore 
# WER on dev(fglarge)              3.34         2.97            2.98
# WER on dev(tglarge)              3.44         3.02            3.07
# WER on dev_other(fglarge)        8.70         7.98            8.00
# WER on dev_other(tglarge)        9.25         8.28            8.35
# WER on test(fglarge)             3.77         3.41            3.40
# WER on test(tglarge)             3.85         3.50            3.47
# WER on test_other(fglarge)       8.91         8.22            8.21
# WER on test_other(tglarge)       9.31         8.55            8.49

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
epochs=20

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

# test of 960 hours training transcriptions
text=data/train_960/text
lexicon=data/lang_nosp/words.txt
text_dir=data/rnnlm/text_960_1a
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
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/librispeech.txt
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
  rnnlm/train_rnnlm.sh --num-jobs-final 2 \
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
  for decode_set in test_clean test_other dev_clean dev_other; do
    for LM in fglarge tglarge; do 
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        # Lattice rescoring
        rnnlm/lmrescore$pruned.sh \
            --cmd "$decode_cmd --mem 8G" \
            --weight 0.45 --max-ngram-order $ngram_order \
            data/lang_test_$LM $dir \
            data/${decode_set}_hires ${decode_dir} \
            exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_${LM}_${decode_dir_suffix}_rescore
    done
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in test_clean test_other dev_clean dev_other; do
    for LM in fglarge tglarge; do 
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        # Nbest rescoring
        rnnlm/lmrescore_nbest.sh \
            --cmd "$decode_cmd --mem 8G" --N 20 \
            0.4 data/lang_test_$LM $dir \
            data/${decode_set}_hires ${decode_dir} \
            exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_${LM}_${decode_dir_suffix}_nbest_rescore
    done
  done
fi

exit 0
