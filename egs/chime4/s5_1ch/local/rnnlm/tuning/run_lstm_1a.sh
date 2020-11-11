#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Szu-Jui Chen

# This script trains LMs on the Chime4 data.
# rnnlm/train_rnnlm.sh: best iteration (out of 120) was 91, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 23.2 / 25.6.
# Train objf: -5.63 -4.52 -4.20 -4.05 -3.96 -3.89 -3.83 -3.79 -3.76 -3.73 -3.70 -3.67 -3.65
# -3.63 -3.61 -3.59 -3.58 -3.56 -3.54 -3.53 -3.52 -3.50 -3.49 -3.48 -3.47 -3.46 -3.45 -3.44
# -3.43 -3.42 -3.43 -3.41 -3.39 -3.38 -3.38 -3.37 -3.35 -3.34 -3.34 -3.33 -3.32 -3.31 -3.31
# -3.30 -3.29 -3.28 -3.28 -3.27 -3.26 -3.25 -3.25 -3.25 -3.23 -3.22 -3.23 -3.22 -3.21 -3.20
# -3.20 -3.19 -3.19 -3.18 -3.18 -3.17 -3.16 -3.15 -3.16 -3.15 -3.14 -3.13 -3.13 -3.13 -3.12
# -3.11 -3.12 -3.11 -3.10 -3.09 -3.09 -3.09 -3.08 -3.07 -3.07 -3.07 -3.06 -3.05 -3.05 -3.05
# -3.04 -3.04 -3.04 -3.03 -3.00 -3.02 -3.00 -2.99 -3.00 -2.99 -2.99 -2.98 -2.96 -2.97 -2.96
# -2.95 -2.96 -2.95 -2.95 -2.94 -2.93 -2.93 -2.92 -2.91 -2.92 -2.91 -2.91 -2.91 -2.89 -2.90 -2.89 -2.88 
#Dev objf:   -11.73 -5.17 -4.46 -4.21 -4.06 -3.96 -3.88 -3.82 -3.79 -3.73 -3.69 -3.68 -3.63 
# -3.61 -3.59 -3.58 -3.54 -3.54 -3.53 -3.51 -3.50 -3.47 -3.47 -3.46 -3.44 -3.44 -3.42 -3.42 
# -3.42 -3.42 -3.40 -3.36 -3.35 -3.35 -3.34 -3.34 -3.34 -3.33 -3.32 -3.32 -3.31 -3.31 -3.31 
# -3.30 -3.29 -3.29 -3.29 -3.28 -3.28 -3.28 -3.27 -3.27 -3.26 -3.27 -3.27 -3.26 -3.25 -3.26 
# -3.26 -3.25 -3.25 -3.25 -3.25 -3.25 -3.25 -3.25 -3.26 -3.25 -3.24 -3.25 -3.25 -3.24 -3.24 
# -3.25 -3.25 -3.24 -3.24 -3.25 -3.26 -3.25 -3.25 -3.24 -3.25 -3.25 -3.24 -3.25 -3.25 -3.25 
# -3.24 -3.26 -3.25 -3.25 -3.25 -3.25 -3.25 -3.25 -3.25 -3.25 -3.26 -3.26 -3.26 -3.26 -3.26 
# -3.27 -3.27 -3.27 -3.27 -3.27 -3.27 -3.27 -3.27 -3.27 -3.27 -3.28 -3.28 -3.28 -3.28 -3.29 -3.29 -3.29 

# Begin configuration section.
affix=1a
dir=exp/rnnlm_lstm_${affix}
enhan=$1
embedding_dim=2048
lstm_rpd=512
lstm_nrpd=512
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
use_backward_model=true
ac_model_dir=exp/chain/tdnn1a_sp
decode_dir_suffix=rnnlm_lstm_${affix}
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially


. cmd.sh
. utils/parse_options.sh

srcdir=data/local/local_lm
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_nosp_${affix}
mkdir -p $dir/config
set -e

for f in $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

#prepare training and dev data
if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cp $srcdir/train.rnn $text_dir/chime4.txt.tmp
  sed -e "s/<RNN_UNK>/<UNK>/g" $text_dir/chime4.txt.tmp > $text_dir/chime4.txt
  rm $text_dir/chime4.txt.tmp
  cp $srcdir/valid.rnn $text_dir/dev.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang_chain/words.txt $dir/config/words.txt
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt
  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
chime4   3   1.0
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

# Train model with forward data(forward model)
if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

# Train another model with reversed data(backward model)
if [ $stage -le 4 ] && $use_backward_model; then
  local/rnnlm/run_lstm_back.sh --embedding-dim $embedding_dim \
    --lstm-rpd $lstm_rpd --lstm-nrpd $lstm_nrpd \
    --affix $affix
fi

# Since lattice-rescoring performs worse but faster than nbest-rescoring,
# we only use it to evaluate how good our forward model is.
LM=5gkn_5k # using the 5-gram lm from run_lmrescore_tdnn.sh
tgtdir=${ac_model_dir}_smbr_lmrescore
if [ $stage -le 5 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  for decode_set in dt05_real dt05_simu et05_real et05_simu; do
    decode_dir=$tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${LM}

    # Lattice rescoring
    rnnlm/lmrescore_pruned.sh \
      --cmd "$train_cmd --mem 2G" \
      --weight 0.8 --max-ngram-order $ngram_order \
      data/lang_test_$LM $dir \
      data/${decode_set}_${enhan}_chunked ${decode_dir} \
      $tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${decode_dir_suffix} &
  done
  
  wait
  
  # calc wers for lattice-rescoring results
  local/chime4_calc_wers.sh $tgtdir ${enhan}_${decode_dir_suffix} \
      $tgtdir/graph_tgpr_5k \
      > $tgtdir/best_wer_${enhan}_${decode_dir_suffix}.result
  head -n 15 $tgtdir/best_wer_${enhan}_${decode_dir_suffix}.result
fi

nbest=100
rnnweight=0.8
if [ $stage -le 6 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in dt05_real dt05_simu et05_real et05_simu; do
    decode_dir=$tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${LM}
    (
    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$train_cmd --mem 2G" --N $nbest \
      $rnnweight data/lang_test_$LM $dir \
      data/${decode_set}_${enhan}_chunked ${decode_dir} \
      $tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}

    if $use_backward_model; then
      rnnlm/lmrescore_nbest_back.sh \
        --cmd "$train_cmd --mem 2G" --N $nbest \
        $rnnweight data/lang_test_$LM ${dir}_back \
        data/${decode_set}_${enhan}_chunked \
        $tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest} \
        $tgtdir/decode_tgpr_5k_${decode_set}_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}_bi
    fi
    ) &
  done
  wait
  # calc wers for nbest-rescoring results
  if $use_backward_model; then
    local/chime4_calc_wers.sh $tgtdir ${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}_bi \
      $tgtdir/graph_tgpr_5k \
      > $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}_bi.result
    head -n 15 $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}_bi.result
  else
    local/chime4_calc_wers.sh $tgtdir ${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest} \
        $tgtdir/graph_tgpr_5k \
        > $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}.result
    head -n 15 $tgtdir/best_wer_${enhan}_${decode_dir_suffix}_w${rnnweight}_n${nbest}.result
  fi
fi

exit 0
