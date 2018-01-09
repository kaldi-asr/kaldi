#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains LMs on the swbd LM-training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 35) was 34, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 41.9 / 50.0.
# Train objf: -5.07 -4.43 -4.25 -4.17 -4.12 -4.07 -4.04 -4.01 -3.99 -3.98 -3.96 -3.94 -3.92 -3.90 -3.88 -3.87 -3.86 -3.85 -3.84 -3.83 -3.82 -3.81 -3.80 -3.79 -3.78 -3.78 -3.77 -3.77 -3.76 -3.75 -3.74 -3.73 -3.73 -3.72 -3.71
# Dev objf:   -10.32 -4.68 -4.43 -4.31 -4.24 -4.19 -4.15 -4.13 -4.10 -4.09 -4.05 -4.03 -4.02 -4.00 -3.99 -3.98 -3.98 -3.97 -3.96 -3.96 -3.95 -3.94 -3.94 -3.94 -3.93 -3.93 -3.93 -3.92 -3.92 -3.92 -3.92 -3.91 -3.91 -3.91 -3.91

# %WER 11.1 | 1831 21395 | 89.9 6.4 3.7 1.0 11.1 46.3 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped/score_13_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 9.9 | 1831 21395 | 91.0 5.8 3.2 0.9 9.9 43.2 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e/score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 9.9 | 1831 21395 | 91.0 5.8 3.2 0.9 9.9 42.9 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e_nbest/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys

# %WER 15.9 | 4459 42989 | 85.7 9.7 4.6 1.6 15.9 51.6 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped/score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 14.4 | 4459 42989 | 87.0 8.7 4.3 1.5 14.4 49.4 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 14.4 | 4459 42989 | 87.1 8.7 4.2 1.5 14.4 49.0 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e_nbest/score_10_0.0/eval2000_hires.ctm.filt.sys

# Begin configuration section.

dir=exp/rnnlm_lstm_1e
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
ac_model_dir=exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp
decode_dir_suffix=rnnlm_1e
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

text=data/train_nodev/text
fisher_text=data/local/lm/fisher/text1.gz
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
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/swbd.txt
  cat > $dir/config/hesitation_mapping.txt <<EOF
hmm hum
mmm um
mm um
mhm um-hum 
EOF
  gunzip -c $fisher_text | awk 'NR==FNR{a[$1]=$2;next}{for (n=1;n<=NF;n++) if ($n in a) $n=a[$n];print $0}' \
    $dir/config/hesitation_mapping.txt - > $text_dir/fisher.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
swbd   3   1.0
fisher   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<unk>,[noise],[laughter],[vocalized-noise]' \
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

LM=sw1_fsh_fg # using the 4-gram const arpa file as old lm
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
#  LM=sw1_tg # if using the original 3-gram G.fst as old lm
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in eval2000; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}_looped

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in eval2000; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}_looped

    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      0.8 data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

exit 0
