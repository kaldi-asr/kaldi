#!/usr/bin/env bash

# Begin configuration section.
# This recipe is based on wsj/s5/local/rnnlm/run_rnnlm.sh

# without RNNLM
# eval1: %WER 10.05 [ 2615 / 26028, 274 ins, 494 del, 1847 sub ] exp/chain/tdnn1b/decode_eval1/wer_10_0.5
# eval2: %WER 8.09 [ 2157 / 26661, 177 ins, 394 del, 1586 sub ] exp/chain/tdnn1b/decode_eval2/wer_12_0.5
# eval3: %WER 9.33 [ 1603 / 17189, 169 ins, 304 del, 1130 sub ] exp/chain/tdnn1b/decode_eval3/wer_13_0.5

# with RNNLM
# eval1: %WER 8.52 [ 2218 / 26028, 239 ins, 454 del, 1525 sub ] exp/chain/tdnn1b/decode_eval1_lstm_rnnlm/wer_12_0.0
# eval2: %WER 7.37 [ 1964 / 26661, 227 ins, 259 del, 1478 sub ] exp/chain/tdnn1b/decode_eval2_lstm_rnnlm/wer_10_0.0
# eval3: %WER 7.83 [ 1346 / 17189, 140 ins, 260 del, 946 sub ] exp/chain/tdnn1b/decode_eval3_lstm_rnnlm/wer_13_0.5

dir=exp/rnnlm_tdnn1b
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=20
stage=0
train_stage=-10

# variables for rnnlm rescoring
ac_model_dir=exp/chain/tdnn1b
ngram_order=4
decode_dir_suffix=lstm_rnnlm

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

text=data/local/lm/train.gz
wordlist=data/lang_csj_tg/words.txt
text_dir=data/rnnlm/text_nosp
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist " && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 500 lines as dev data.
  gunzip -c $text | awk -v text_dir=$text_dir '{if(NR%500 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/csj.txt
fi

if [ $stage -le 1 ]; then
  # the training scripts require that <s>, </s> and <brk> be present in a particular
  # order.
  cp $wordlist $dir/config/ 
  n=$(wc -l $dir/config/words.txt | cut -d " " -f 1)
  echo "<brk> $n" >> $dir/config/words.txt 
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
csj   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=50000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>,<sp>' \
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
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

if [ $stage -le 4 ]; then
  for decode_set in eval1 eval2 eval3; do
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_csj_tg $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix} &
  done
  wait
fi

exit 0
