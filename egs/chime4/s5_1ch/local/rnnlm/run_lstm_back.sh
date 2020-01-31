#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Szu-Jui Chen

# This script trains LMs on the reversed Chime4 data, which we
# call it backward model.

# Begin configuration section.
affix=1a
dir=exp/rnnlm_lstm_${affix}_back
embedding_dim=2048
lstm_rpd=512
lstm_nrpd=512
stage=-10
train_stage=-10

# variables for lattice rescoring
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially

. cmd.sh
. utils/parse_options.sh

srcdir=data/local/local_lm
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_nosp_${affix}_back
mkdir -p $dir/config
set -e

for f in $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

#prepare training and dev data
if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $srcdir/train.rnn | awk '{for(i=NF;i>0;i--) printf("%s ",$i); print""}'> $text_dir/chime4.txt.tmp
  sed -e "s/<RNN_UNK>/<UNK>/g" $text_dir/chime4.txt.tmp > $text_dir/chime4.txt
  rm $text_dir/chime4.txt.tmp
  cat $srcdir/valid.rnn | awk '{for(i=NF;i>0;i--) printf("%s ",$i); print""}'> $text_dir/dev.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang_chain/words.txt $dir/config/words.txt
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt
  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
chime4   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<UNK>,<brk>' \
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
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

exit 0
