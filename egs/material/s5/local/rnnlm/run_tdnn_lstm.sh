#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2017  Hainan Xu

# This script trains LMs on the swbd LM-training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 3, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 105.1 / 223.6.
# Train objf: -5.72 -5.28 -4.92 -4.64 -4.36 -4.09 -3.85 -3.62 -3.40 -3.23 
# Dev objf:   -9.99 -5.71 -5.43 -5.41 -5.52 -5.69 -5.86 -6.09 -6.29 -6.49 

# %WER 39.14 [ 24322 / 62144, 3199 ins, 6127 del, 14996 sub ] exp/chain/tdnn_lstm1a_tree6000_sp_ld5/decode_dev/wer_10_0.0
# %WER 37.60 [ 23367 / 62144, 3129 ins, 5918 del, 14320 sub ] exp/chain/tdnn_lstm1a_tree6000_sp_ld5/decode_dev_rnnlm_1a/wer_9_0.5

# Begin configuration section.

dir=exp/rnnlm_lstm_1a
embedding_dim=512
lstm_rpd=128
lstm_nrpd=128
stage=-10
train_stage=-10

# variables for lattice rescoring
run_rescore=true
ac_model_dir=exp/chain/tdnn_lstm1a_sp_ld5
decode_dir_suffix=rnnlm_1a
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

text=data/train/text
lexicon=data/local/dict_nosp/lexiconp.txt
text_dir=data/rnnlm/text_nosp_1e
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/train.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   10   1.0
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
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 --embedding_l2 0.01 \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

LM=test
if [ $stage -le 4 ] && $run_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in dev; do
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}
  done
fi

exit 0
