#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Szu-Jui Chen

# This script trains LMs on the Chime4 data.

# Begin configuration section.

dir=exp/rnnlm_lstm_1a
enhan=$1
embedding_dim=2048
lstm_rpd=512
lstm_nrpd=512
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
ac_model_dir=exp/chain/tdnn1b_sp
decode_dir_suffix=rnnlm_1a
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially


. cmd.sh
. utils/parse_options.sh

tgtdir=data/local/local_lm
fisher_text=data/local/lm/fisher/text1.gz
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_nosp_1a
mkdir -p $dir/config
set -e

for f in $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

#prepare training and dev data
if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $tgtdir/train.rnn | uniq >$text_dir/chime4.txt
  cat $tgtdir/valid.rnn | uniq >$text_dir/dev.txt
  zcat $fisher_text > $text_dir/fisher.txt
fi

if [ $stage -le 1 ]; then
  cat $tgtdir/vocab_5k.rnn | awk '{printf("%s %d\n", $1, NR)}' | sed '1s/^/<eps> 0\n/' > $dir/config/words.txt
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt
  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<RNN_UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
chime4   3   1.0
fisher 1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<RNN_UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<RNN_UNK>,<brk>' \
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

LM=tgpr_5k # using the 3-gram as old lm
tgtdir=${ac_model_dir}_smbr_lmrescore
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  for decode_set in dt05_real dt05_simu et05_real et05_simu; do
    decode_dir=${ac_model_dir}/decode_${LM}_${decode_set}_${enhan}

    # Lattice rescoring
    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_test_$LM $dir \
      data/${decode_set}_${enhan}_hires ${decode_dir} \
      $tgtdir/decode_${LM}_${decode_set}_${enhan}_${decode_dir_suffix}
  done
  # calc wers for lattice-rescoring results
  local/chime4_calc_wers.sh $tgtdir ${enhan}_${decode_dir_suffix} \
      $tgtdir/graph_tgpr_5k \
      > $tgtdir/best_wer_${enhan}_${decode_dir_suffix}.result
  head -n 15 $tgtdir/best_wer_${enhan}_${decode_dir_suffix}.result
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in dt05_real dt05_simu et05_real et05_simu; do
    decode_dir=${ac_model_dir}/decode_${LM}_${decode_set}_${enhan}

    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 50 \
      0.8 data/lang_test_$LM $dir \
      data/${decode_set}_${enhan}_hires ${decode_dir} \
      $tgtdir/decode_${LM}_${decode_set}_${enhan}_${decode_dir_suffix}_nbest
  done
  # calc wers for nbest-rescoring results
  local/chime4_calc_wers.sh $tgtdir ${enhan}_${decode_dir_suffix}_nbest \
      $tgtdir/graph_tgpr_5k \
      > $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_nbest.result
  head -n 15 $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_nbest.result
fi

exit 0
