#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#                2017  Hainan Xu
#                2018  Ke Li
#                2018  Yiming Wang


# [for swahili]
# rnnlm/train_rnnlm.sh: best iteration (out of 40) was 38, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 140.6 / 1019.4.
# Train objf: -6.28 -5.90 -5.70 -5.56 -5.47 -5.40 -5.34 -5.29 -5.25 -5.22 -5.17 -5.16 -5.13 -5.10 -5.07 -5.06 -5.04 -5.01 -4.99 -4.98 -4.97 -4.96 -4.93 -4.93 -4.91 -4.91 -4.89 -4.88 -4.87 -4.86 -4.84 -4.85 -4.81 -4.79 -4.78 -4.76 -4.75 -4.74 -4.73
# Dev objf:   -8.69 -7.76 -7.31 -7.03 -6.98 -7.00 -6.96 -6.96 -6.93 -6.94

# %WER 36.75 [ 22836 / 62144, 2758 ins, 6307 del, 13771 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore/wer_10_0.0
# %WER 38.91 [ 24181 / 62144, 2750 ins, 6579 del, 14852 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore_nbest/wer_10_0.0
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# | Sum/Avg                            |  9906   59164  | 62.2     23.8    14.0     3.5     41.3    49.1  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# |  Sum/Avg                            |  9906    59164  |  61.9     23.6     14.6      3.2     41.4     49.5  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# | Sum/Avg                            |  5322   37120  | 66.2     21.2    12.6     2.9     36.8    49.8  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# |  Sum/Avg                            |  5322    37120  |  65.8     21.1     13.1      2.7     36.9     49.9  |

# [for tagalog]
# rnnlm/train_rnnlm.sh: best iteration (out of 320) was 125, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 141.2 / 259.6.
# Train objf: -6.08 -5.78 -5.62 -5.52 -5.45 -5.40 -5.36 -5.32 -5.28 -5.26 -5.23 -5.20 -5.18 -5.16 -5.14 -5.13 -5.11 -5.10 -5.09 -5.07 -5.06 -5.05 -5.03 -5.03 -5.02 -5.01 -5.00 -4.99 -4.99 -4.97 -4.97 -4.97 -4.96 -4.94 -4.94 -4.93 -4.93 -4.92 -4.91 -4.92 -4.91 -4.90 -4.89 -4.89 -4.89 -4.88 -4.87 -4.87 -4.87 -4.86 -4.86 -4.85 -4.85 -4.84 -4.84 -4.84 -4.83 -4.83 -4.83 -4.82 -4.82 -4.82 -4.81 -4.82 -4.81 -4.81 -4.80 -4.79 -4.79 -4.79 -4.79 -4.80 -4.79 -4.79 -4.79 -4.80 -4.79 -4.78 -4.78 -4.79 -4.77 -4.79 -4.79 -4.78 -4.78 -4.78 -4.78 -4.78 -4.77 -4.77 -4.79 -4.79 -4.78 -4.78 -4.78 -4.78 -4.78 -4.79 -4.78 -4.80 -4.79 -4.78 -4.79 -4.80 -4.80 -4.79 -4.79 -4.77 -4.78 -4.77 -4.77 -4.78 -4.75 -4.80 -4.78 -4.77 -4.76 -4.77 -4.76 -4.76 -4.75 -4.75 -4.76 -4.76 -4.77 -4.75 -4.75 -4.75 -4.76 -4.75 -4.76 -4.74 -4.75 -4.75 -4.76 -4.75 -4.75 -4.75 -4.74 -4.76 -4.75 -4.74 -4.78 -4.74 -4.73 -4.77 -4.76 -4.75 -4.74 -4.73 -4.73 -4.75 -4.75 -4.74 -4.76 -4.73 -4.72 -4.76 -4.72 -4.72 -4.73 -4.72 -4.73 -4.75 -4.72 -4.73 -4.76 -4.75 -4.72 -4.72 -4.74 -4.75 -4.73 -4.72 -4.74 -4.74 -4.73 -4.74 -4.74 -4.74 -4.72 -4.70 -4.72 -4.75 -4.74 -4.75 -4.74 -4.76 -4.72 -4.72 -4.74 -4.75 -4.71 -4.74 -4.73 -4.73 -4.73 -4.73 -4.74 -4.75 -4.73 -4.73 -4.72 -4.71 -4.72 -4.71 -4.72 -4.75 -4.72 -4.71 -4.74 -4.71 -4.70 -4.73 -4.73 -4.75 -4.75 -4.72 -4.72 -4.73 -4.75 -4.73 -4.72 -4.72 -4.72 -4.73 -4.76 -4.73 -4.76 -4.74 -4.73 -4.74 -4.74 -4.74 -4.73 -4.73 -4.73 -4.70 -4.73 -4.74 -4.72 -4.73 -4.73 -4.75 -4.72 -4.73 -4.73 -4.75 -4.73 -4.75 -4.75 -4.73 -4.75 -4.74 -4.75 -4.77 -4.74 -4.75 -4.74 -4.73 -4.77 -4.75 -4.74 -4.75 -4.74 -4.77 -4.76 -4.75 -4.79 -4.78 -4.76 -4.76 -4.77 -4.76 -4.75 -4.74 -4.74 -4.78 -4.77 -4.77 -4.78 -4.79 -4.79 -4.79 -4.76 -4.77 -4.76 -4.79 -4.76 -4.77 -4.76 -4.78 -4.80 -4.79 -4.78 -4.82 -4.82 -4.79 -4.80 -4.81 -4.79 -4.77 -4.79 -4.82 -4.81 -4.82 -4.83 -4.85 -4.84 -4.83 -4.85 -4.88 -4.85 -4.87 -4.86 -4.84 -4.87 -4.85 -4.84 
# Dev objf:   -8.70 -7.03 -60340.00 -6.61 -6.45 -6.54 -60340.00 -6.34 -60340.00 -60340.00 -6.15 -6.12 -6.03 -6.03 -60340.00 -60340.00 -6.64 -60340.00 -6.01 -5.91 -5.93 -6.06 -5.92 -5.95 -6.00 -6.17 -6.06 -5.92 -5.92 -60340.00 -6.03 -5.93 -5.98 -60340.00 -6.00 -5.90 -5.84 -6.00 -60340.00 -5.95 -5.89 -60340.00 -5.90 -6.14 -5.84 -5.92 -5.83 -5.86 -5.89 -5.84 -60340.00 -5.90 -5.80 -5.87 -5.87 -60340.00 -5.79 -60340.00 -60340.00 -60340.00 -6.56 -5.88 -5.94 -60340.00 -5.84 -60340.00 -5.84 -5.81 -5.77 -60340.00 -60340.00 -60340.00 -5.81 -5.90 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.72 -5.79 -60340.00 -60340.00 -60340.00 -60340.00 -5.72 -5.80 -60340.00 -60340.00 -5.68 -5.73 -5.74 -60340.00 -5.67 -5.63 -60340.00 -5.75 -60340.00 -5.66 -5.71 -5.73 -5.73 -5.75 -60340.00 -5.77 -60340.00 -5.70 -5.70 -5.82 -60340.00 -60340.00 -5.77 -5.72 -5.75 -60340.00 -5.56 -60340.00 -5.73 -60340.00 -60340.00 -5.99 -5.77 -60340.00 -5.65 -5.80 -60340.00 -60340.00 -5.64 -5.67 -5.73 -5.59 -60340.00 -60340.00 -5.73 -60340.00 -60340.00 -5.83 -5.58 -5.64 -5.75 -60340.00 -5.77 -5.68 -60340.00 -60340.00 -5.70 -5.85 -60340.00 -60340.00 -5.82 -6.15 -5.74 -5.73 -5.75 -60340.00 -60340.00 -5.86 -60340.00 -5.80 -5.79 -5.81 -60340.00 -5.89 -60340.00 -5.81 -5.71 -60340.00 -60340.00 -5.65 -5.87 -60340.00 -60340.00 -60340.00 -5.83 -60340.00 -5.94 -5.74 -5.75 -5.75 -60340.00 -5.76 -5.73 -5.76 -60340.00 -60340.00 -5.85 -5.91 -5.98 -60340.00 -5.88 -5.86 -60340.00 -60340.00 -60340.00 -60340.00 -5.91 -5.81 -5.86 -60340.00 -6.10 -6.17 -60340.00 -60340.00 -5.82 -5.82 -60340.00 -60340.00 -6.78 -5.71 -5.87 -60340.00 -60340.00 -5.98 -5.94 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.81 -60340.00 -60340.00 -60340.00 -5.74 -60340.00 -5.83 -60340.00 -5.96 -5.80 -60340.00 -60340.00 -60340.00 -5.82 -60340.00 -60340.00 -60340.00 -60340.00 -5.80 -60340.00 -60340.00 -60340.00 -60340.00 -5.79 -60340.00 -6.13 -5.97 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.97 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.98 -60340.00 -60340.00 -60340.00 -5.85 -5.92 -5.85 -5.82 -6.04 -60340.00 -60340.00 -60340.00 -60340.00 -5.93 -60340.00 -5.85 -5.87 -5.77 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.89 -60340.00 -60340.00 -60340.00 -60340.00 -6.18 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -60340.00 -5.92 -6.01

# %WER 46.07 [ 29664 / 64382, 3133 ins, 9896 del, 16635 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore/wer_10_0.5
# %WER 47.47 [ 30563 / 64382, 3568 ins, 8934 del, 18061 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore_nbest/wer_10_0.5
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# | Sum/Avg                            | 10551   87329  | 53.7     25.3    21.0     4.6     51.0    65.6  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# |  Sum/Avg                            | 10551    87329  |  53.4     24.9     21.6      4.3     50.9     65.6  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# | Sum/Avg                            |  5933   56887  | 52.6     25.0    22.4     4.9     52.3    73.8  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# |  Sum/Avg                            |  5933    56887  |  52.3     24.5     23.1      4.5     52.2     73.9  |

# [for somali]
# rnnlm/train_rnnlm.sh: best iteration (out of 800) was 133, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 414.5 / 860.9.

# %WER 56.54 [ 46160 / 81637, 4654 ins, 13070 del, 28436 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore/wer_10_0.0
# %WER 57.85 [ 47226 / 81637, 5002 ins, 12287 del, 29937 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore_nbest/wer_10_0.0
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# | Sum/Avg                            |  9852   90609  | 50.4     33.3    16.3     8.2     57.8    74.8  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis1_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis1_segmented_hires.ctm.sys 
# |  Sum/Avg                            |  9852    90609  |  50.4     33.2     16.4      8.1     57.7     74.9  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# | Sum/Avg                            |  8275   67640  | 53.0     32.8    14.2     8.5     55.5    69.3  |
# grep 'Sum' exp/chain/tdnn1b_sp/decode_analysis2_segmented_rnnlm_rescore_nbest/score_10_0.0/analysis2_segmented_hires.ctm.sys 
# |  Sum/Avg                            |  8275    67640  |  53.0     32.7     14.3      8.3     55.3     69.2  |


# Begin configuration section.

embedding_dim=512
lstm_rpd=128
lstm_nrpd=128
stage=0
train_stage=-10
epochs=40

# variables for lattice rescoring
run_rescore=true
decode_dir_suffix=rnnlm
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

ac_model_dir=exp/chain/tdnn1b_sp
decode_sets="dev analysis1_segmented analysis2_segmented test_dev_segmented eval1_segmented eval2_segmented eval3_segmented"

dir=exp/rnnlm_lstm_1a
text_dir=data/rnnlm/text
train_text=data/lm/train.txt
dev_text=data/lm/dev.txt
bitext=data/bitext/text.txt
monotext=data/mono/text.txt

lang=data/lang_combined_chain
tree_dir=exp/chain/tree_sp

. ./cmd.sh
. ./utils/parse_options.sh


mkdir -p $dir/config
set -e

for f in ${train_text} ${dev_text} $bitext $monotext; do

  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; look at stage 12 in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train_text > $text_dir/train.txt
  cat $dev_text > $text_dir/dev.txt
  cat $bitext > $text_dir/bitext.txt
  cat $monotext > $text_dir/monotext.txt

fi

if [ $stage -le 1 ]; then
  cp $lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   1   1.0
bitext  1   1.0
monotext  1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<unk>,<noise>,<spnoise>,<sil>' \
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
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 --embedding-l2 0.001 \
                  --stage $train_stage --num-epochs $epochs --cmd "$train_cmd" $dir
fi

LM=combined_chain
if [ $stage -le 4 ] && $run_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  rm $dir/.error 2>/dev/null || true
  for decode_set in ${decode_sets}; do
    (
      decode_dir=${ac_model_dir}/decode_${decode_set}
      skip_scoring=false
      if [ ${decode_set} != "dev" ]; then skip_scoring=true; fi

      # Lattice rescoring
      rnnlm/lmrescore$pruned.sh \
        --cmd "$decode_cmd" \
        --weight 0.5 --max-ngram-order $ngram_order --max-arcs 20000 \
        --skip-scoring ${skip_scoring} \
        data/lang_$LM $dir data/${decode_set}_hires \
        ${decode_dir} ${decode_dir}_${decode_dir_suffix}_rescore || exit 1

      if [ ${decode_set} != "dev" ]; then
        local/postprocess_test.sh ${decode_set} ${tree_dir}/graph_combined \
          ${decode_dir}_${decode_dir_suffix}_rescore
      fi
    ) || touch $dir/.error &
  done
fi
wait
#[ -f $dir/.error ] && echo "$0: there was a problem while rescoring" && exit 1

if [ $stage -le 5 ]; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"

  rm $dir/.error 2>/dev/null || true
  for decode_set in ${decode_sets}; do
    (
      decode_dir=${ac_model_dir}/decode_${decode_set}
      skip_scoring=false
      if [ ${decode_set} != "dev" ]; then skip_scoring=true; fi

      # Lattice rescoring
      rnnlm/lmrescore_nbest.sh \
        --N 20 \
        --cmd "$decode_cmd" \
        --skip-scoring ${skip_scoring} \
        0.5 data/lang_$LM $dir data/${decode_set}_hires \
        ${decode_dir}_${decode_dir_suffix}_rescore ${decode_dir}_${decode_dir_suffix}_rescore_nbest || exit 1

      if [ ${decode_set} != "dev" ]; then
        local/postprocess_test.sh ${decode_set} ${tree_dir}/graph_combined \
          ${decode_dir}_${decode_dir_suffix}_rescore_nbest
      fi
    ) || touch $dir/.error 
  done
fi

exit 0
