#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson
#           2017  Hainan Xu
#           2017  Ke Li
#           2018  François Hernandez (Ubiqus)
#
# rnnlm/train_rnnlm.sh: best iteration (out of 1060) was 1050, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 90.0 / 92.0.

# System                    tdnn_1a   tdnnf_1a
# WER on dev(orig)            8.2       7.9
# WER on dev(ngram)           7.6       7.2
# WER on dev(rnnlm)           6.3       6.1
# WER on test(orig)           8.1       8.0
# WER on test(ngram)          7.7       7.5
# WER on test(rnnlm)          6.7       6.6

# Begin configuration section.
dir=exp/rnnlm_lstm_tdnn_a
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
stage=-10
train_stage=-10
epochs=20

. ./cmd.sh
. utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd

text_from_audio=data/train/text
wordlist=data/lang_chain/words.txt
dev_sents=10000
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/prepare_data.sh and utils/prepare_lang.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  gunzip -c db/TEDLIUM_release-3/LM/*.en.gz | sed 's/ <\/s>//g' > $text_dir/train.txt
  # shuffle text from audio and lm
  cat $text_from_audio | cut -d ' ' -f2- | cat $text_dir/train.txt |\
    shuf > data/rnnlm/full_lm_data.shuffled
  # create dev and train sets based on audio and LM data
  cat data/rnnlm/full_lm_data.shuffled | head -n $dev_sents> $text_dir/dev.txt
  cat data/rnnlm/full_lm_data.shuffled | tail -n +$[$dev_sents+1] > $text_dir/ted.txt

fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
ted   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=10000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>' \
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
echo "rnnlm dir done"

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi


exit 0
