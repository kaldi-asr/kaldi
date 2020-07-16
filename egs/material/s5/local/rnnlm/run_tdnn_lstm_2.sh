#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#                2017  Hainan Xu
#                2018  Ke Li
#                2018  Yiming Wang


# [for swahili]
# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 5, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 59.1 / 273.1.
# Train objf: -5.48 -4.75 -4.47 -4.30 -4.17 -4.06 -3.96 -3.87 -3.77 -3.68 
# Dev objf:   -10.79 -6.00 -5.75 -5.69 -5.62 -5.61 -5.62 -5.66 -5.66

# %WER 35.84 [ 22270 / 62144, 2573 ins, 6961 del, 12736 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore/wer_11_0.5
# %WER 48.49 [ 28692 / 59166, 2310 ins, 9200 del, 17182 sub ] exp/chain/tdnn1b_sp/decode_analysis1_segmented_reseg_rnnlm_rescore

# [for tagalog]
# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 4, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 73.6 / 106.2.
# Train objf: -5.55 -4.83 -4.58 -4.41 -4.28 -4.17 -4.06 -3.96 -3.86
# Dev objf:   -10.54 -4.87 -4.72 -4.67 -4.67 -4.69 -4.71 -4.74 -4.78

# %WER 42.91 [ 27628 / 64382, 3624 ins, 8301 del, 15703 sub ] exp/chain/tdnn1b_sp/decode_dev_rnnlm_rescore/wer_10_0.0
# %WER 55.55 [ 48530 / 87362, 4030 ins, 19326 del, 25174 sub ] exp/chain/tdnn1b_sp/decode_analysis1_segmented_reseg_rnnlm_rescore

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
#decode_sets="dev analysis1_segmented_reseg test_dev_segmented_reseg eval1_segmented_reseg eval2_segmented_reseg"
decode_sets="dev analysis1_segmented test_dev_segmented eval1_segmented eval2_segmented eval3_segmented"
decode_sets="analysis2_segmented"
#decode_sets="dev eval1_segmented eval2_segmented"
dir=exp/rnnlm_lstm_1a
text_dir=data/rnnlm/text
train_text=data/lm/train.txt
dev_text=data/lm/dev.txt
bitext=data/bitext/text.txt
lang=data/lang_combined_chain
tree_dir=exp/chain/tree_sp

. ./cmd.sh
. ./utils/parse_options.sh


mkdir -p $dir/config
set -e

for f in ${train_text} ${dev_text} $bitext; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; look at stage 12 in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train_text > $text_dir/train.txt
  cat $dev_text > $text_dir/dev.txt
  cat $bitext > $text_dir/bitext.txt
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
