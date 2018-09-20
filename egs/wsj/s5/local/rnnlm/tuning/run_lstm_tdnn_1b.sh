#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# rnnlm/train_rnnlm.sh: best iteration (out of 80) was 79, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 44.3 / 49.9. 
# Train objf: -1038.00 -5.35 -5.04 -4.87 -4.76 -4.68 -4.61 -4.56 -4.52 -4.47 -4.44 -4.41 -4.37 -4.35 -4.33 -4.31 -4.29 -4.27 -4.25 -4.24 -4.23 -4.21 -4.19 -4.17 -4.16 -4.15 -4.13 -4.12 -4.11 -4.10 -4.09 -4.07 -4.07 -4.06 -4.05 -4.04 -4.03 -4.02 -4.01 -4.00 -3.99 -3.98 -3.98 -3.97 -3.96 -3.96 -3.95 -3.94 -3.93 -3.93 -3.92 -3.92 -3.91 -3.91 -3.90 -3.90 -3.89 -3.88 -3.88 -3.88 -3.88 -3.88 -3.86 -3.86 -3.85 -3.85 -3.84 -3.83 -3.83 -3.83 -3.82 -3.82 -3.81 -3.81 -3.80 -3.80 -3.79 -3.79 -3.79 -3.79 
# Dev objf:   -11.73 -5.66 -5.18 -4.96 -4.82 -4.73 -4.66 -4.59 -4.54 -4.51 -4.47 -4.44 -4.40 -4.38 -4.36 -4.34 -4.32 -4.30 -4.28 -4.27 -4.26 -4.21 -4.19 -4.18 -4.16 -4.15 -4.14 -4.13 -4.12 -4.12 -4.11 -4.09 -4.09 -4.08 -4.07 -4.07 -4.06 -4.06 -4.05 -4.04 -4.04 -4.04 -4.03 -4.02 -4.02 -4.01 -4.01 -4.00 -4.00 -4.00 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.97 -3.97 -3.97 -3.97 -3.96 -3.95 -3.95 -3.94 -3.94 -3.94 -3.94 -3.93 -3.93 -3.93 -3.93 -3.93 -3.93 -3.92 -3.92 -3.92 -3.92 -3.92 -3.91 -3.91 

# WER numbers

# without RNNLM
# %WER 7.51 [ 618 / 8234, 82 ins, 112 del, 424 sub ] exp/chain/tdnn_lstm1b_sp/decode_looped_tgpr_dev93/wer_10_1.0
# %WER 5.21 [ 294 / 5643, 55 ins, 34 del, 205 sub ] exp/chain/tdnn_lstm1b_sp/decode_looped_tgpr_eval92/wer_11_0.5

# with RNNLM
# %WER 5.74 [ 473 / 8234, 81 ins, 76 del, 316 sub ] exp/chain/tdnn_lstm1b_sp/decode_looped_tgpr_dev93_rnnlm/wer_14_1.0
# %WER 4.27 [ 241 / 5643, 62 ins, 23 del, 156 sub ] exp/chain/tdnn_lstm1b_sp/decode_looped_tgpr_eval92_rnnlm/wer_12_1.0

# Begin configuration section.

dir=exp/rnnlm_lstm_tdnn_1b
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=20
stage=-10
train_stage=-10

# variables for rnnlm rescoring
ac_model_dir=exp/chain/tdnn_lstm1b_sp
ngram_order=4
decode_dir_suffix=rnnlm

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd


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
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

LM=tgpr
if [ $stage -le 4 ]; then
  for decode_set in dev93 eval92; do
    decode_dir=${ac_model_dir}/decode_looped_${LM}_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.8 --max-ngram-order $ngram_order \
      data/lang_test_$LM $dir \
      data/test_${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix} &
  done
  wait
fi

exit 0
