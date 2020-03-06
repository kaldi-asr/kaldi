#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2018  Ke Li


# Begin configuration section.

dir=exp/rnnlm_lstm_1a
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10
epochs=4

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false
ac_model_dir=exp/chain_cleanup/tdnn_1d_sp
decode_dir_suffix=rnnlm_1a
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true
. path.sh
. ./cmd.sh
. ./utils/parse_options.sh

text=data/local/lm_large_4gram/train_text.gz
lexicon=data/lang_large_test/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e
for f in $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for run.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 2000 lines as dev data.
  gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/mandarin.txt
fi

if [ $stage -le 1 ]; then
  cp $lexicon $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
mandarin   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --top-word-features=5000 \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<UNK>,[VOCALIZED-NOISE],[NOISE],[LAUGHTER]' \
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
  rnnlm/train_rnnlm.sh --num-jobs-final 8 \
                       --stage $train_stage \
                       --num-epochs $epochs \
                       --cmd "$train_cmd" $dir
fi

echo "RNNLM training finished"
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
#  LM=tgsmall # if using the original 3-gram G.fst as old lm
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  LM="large_test"
  for decode_set in dev eval; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
        --cmd "$decode_cmd --mem 8G" \
        --weight 0.45 --max-ngram-order $ngram_order \
        data/lang_${LM} $dir \
        data/${decode_set}_hires ${decode_dir} \
        $ac_model_dir/decode_${decode_set}_${LM}_${decode_dir_suffix}_rescore
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  LM="large_test"
  for decode_set in dev eval; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
    # Nbest rescoring
    rnnlm/lmrescore_nbest.sh \
        --cmd "$decode_cmd --mem 8G" --N 20 \
        0.4 data/lang_${LM} $dir \
        data/${decode_set}_hires ${decode_dir} \
        $ac_model_dir/decode_${decode_set}_${LM}_${decode_dir_suffix}_nbest_rescore
  done
fi

exit 0
