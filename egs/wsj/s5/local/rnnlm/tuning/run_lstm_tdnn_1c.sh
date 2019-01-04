#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# This is the same as 1b, with two exceptions: TDNN is used as the acoustic model,
# and the baseline is big-dict 4-gram LM.

# (Manually rename chain dirs for lstm_tdnn_1b and lstm_tdnn_1c as 
# chain1b and chain1c)
# local/chain/compare_wer.sh exp/chain1b/tdnn1g_sp exp/chain1c/tdnn1g_sp
# System                   tdnn1g_sp (1b) tdnn1g_sp (1c)
#WER dev93 (tgpr)                6.72      6.62
#WER dev93 (tg)                  6.46      6.39
#WER dev93 (big-dict,tgpr)       4.76      4.64
#WER dev93 (big-dict,fg)         4.24      4.21
#WER eval92 (tgpr)               4.75      4.73
#WER eval92 (tg)                 4.41      4.32
#WER eval92 (big-dict,tgpr)      2.73      2.69
#WER eval92 (big-dict,fg)        2.50      2.29
# Final train prob        -0.0415   -0.0399
# Final valid prob        -0.0490   -0.0489
# Final train prob (xent)   -0.6449   -0.6345
# Final valid prob (xent)   -0.7038   -0.6937
# Num-params                 8367760   8352320

# Begin configuration section.

dir=exp/rnnlm_lstm_tdnn_1c
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
ac_model_dir=exp/chain/tdnn1g_sp
ngram_order=4
decode_dir_suffix=rnnlm

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd


text=data/local/dict_nosp_larger/cleaned.gz
wordlist=data/lang_nosp_bd/words.txt
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
      data/lang_test_bd_$LM $dir \
      data/test_${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix} &
  done
  wait
fi

exit 0
