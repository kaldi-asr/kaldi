#!/usr/bin/env bash

# Copyright 2020  Guoguo Chen
#           2021  Xiaomi Corporation (Author: Yongqing Wang)
#           2021  ASLP, NWPU (Author: Hang Lyu)
# This script is copied from gigaspeech and modified.
# This script trains LMs on the whole LM-training data.

set -e -o pipefail

# Begin configuration section.
stage=0
train_stage=-10

dir=exp/rnnlm
ac_model_dir=
lang=data/lang_test
test_sets="dev test_meeting test_net"
text=data/corpus/lm_text
decode_iter=

num_epoch=10
num_jobs_initial=1
num_jobs_final=3
words_per_split=400000
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256

# variables for lattice rescoring
run_lat_rescore=true
decode_dir_suffix=rnnlm
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

wordlist=$lang/words.txt
text_dir=data/rnnlm/text
mkdir -p $dir/config

for f in $text $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist." && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 2500 lines as dev data.
  cat $text | sed 's/\t/ /g' | sed 's/[ ][ ]*/ /g' | cut -d ' ' -f2- \
    | awk -v text_dir=$text_dir '{if(NR%100 == 0) { print >text_dir"/dev.txt"; } else {print;}}' \
    >$text_dir/train.txt
fi

if [ $stage -le 1 ]; then
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # Words that are not present in words.txt but are in the training or dev data,
  # will be mapped to <UNK> during training.
  echo "<UNK>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<UNK>,<eps>,!SIL' \
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
  rnnlm/prepare_rnnlm_dir.sh --words-per-split $words_per_split $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  # Note: use '--use-gpu-for-diagnostics', during diagnostic period, you may
  # need to set the '--num-chunks-per-minibatch' option for your rnnlm-get-egs.
  rnnlm/train_rnnlm.sh \
    --use-gpu-for-diagnostics true \
    --num-jobs-initial $num_jobs_initial \
    --num-jobs-final $num_jobs_final \
    --stage $train_stage \
    --num-epochs $num_epoch \
    --cmd "$train_cmd" \
    $dir || exit 1
fi

if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in $test_sets; do
    decode_dir=${ac_model_dir}/decode_${decode_set}${decode_iter:+_$decode_iter}
   (
    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd" \
      --weight 0.45 \
      --max-ngram-order $ngram_order \
      $lang $dir \
      data/${decode_set} ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix} || exit 1
   )
  done
fi

exit 0
