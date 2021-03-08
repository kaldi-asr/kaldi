#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang
#           2020  Ke Li

# This script trains LSTM based LMs on transcription.

# Begin configuration section.

# 1b is emb-dim = 512 and lstm_rpd and lstm_nrpd = 128
#rnnlm/train_rnnlm.sh: best iteration (out of 6) was 2, linking it to final iteration.
#rnnlm/train_rnnlm.sh: train/dev perplexity was 63.1 / 147.1.
#Train objf: -4.41 -4.14 -3.90 -3.68 -3.47
#Dev objf:   -5.09 -4.99 -5.04 -5.21 -5.40
# 1c emb-dim = 256 lstm_rpd = 64 lstm_nrpd = 64 all l2=0
#rnnlm/train_rnnlm.sh: best iteration (out of 6) was 2, linking it to final iteration.
#rnnlm/train_rnnlm.sh: train/dev perplexity was 75.0 / 153.5.
#Train objf: -4.56 -4.31 -4.15 -4.01 -3.87
#Dev objf:   -5.21 -5.03 -5.04 -5.11 -5.22
dir=exp/rnnlm_lstm_1b
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
embedding_l2=0.003 # embedding layer l2 regularize
comp_l2=0.003 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false

# ac_model_dir=exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_cnn_sp
ac_model_dir=exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_cnn_l2_03_4500_ep6_sp
decode_dir_suffix=rnnlm_1b
enhancement=gss_multiarray
chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

train_text=data/train_worn/text
dev_text=data/dev_worn/text
text_dir=data/rnnlm/text
mkdir -p $dir/config
set -e

for f in $train_text $dev_text; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  cat $train_text | cut -d ' ' -f2- > $text_dir/train.txt
  cat $dev_text | cut -d ' ' -f2- > $text_dir/dev.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <unk> during training.
  echo "<unk>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<unk>,[inaudible],[noise],[laughs]' \
                           $dir/config/words.txt > $dir/config/features.txt

lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-3))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-3))
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs 60 --cmd "$train_cmd" $dir
fi

# old 3-gram LM is data/lang/G.fst 
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in dev_gss_multiarray eval_gss_multiarray; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_2stage

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.45 --max-ngram-order $ngram_order \
      data/lang $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_0.45
  done
fi

if [ $stage -le 5 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh --enhancement $enhancement --json $json_dir \
      --dev ${ac_model_dir}/decode_dev_${enhancement}_2stage_${decode_dir_suffix}_0.45 \
      --eval ${ac_model_dir}/decode_eval_${enhancement}_2stage_${decode_dir_suffix}_0.45
fi
exit

if [ $stage -le 6 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in dev_gss_multiarray eval_gss_multiarray; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_2stage

    # Nbest rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      0.5 data/lang $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

# running backward RNNLM, which further improves WERS by combining backward with
# the forward RNNLM trained in this script.
if [ $stage -le 7 ] && $run_backward_rnnlm; then
  local/rnnlm/run_tdnn_lstm_back.sh
fi

exit 0
