#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_1a
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
epochs=20
stage=-10
train_stage=-10

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
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
