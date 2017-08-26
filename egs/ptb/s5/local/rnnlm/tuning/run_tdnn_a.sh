#!/bin/bash

# To be run from the directory egs/ptb/s5.
# This is to be done after local/prepare_rnnlm_data.sh.

# some of the output follows.  This is overfitting badly; model probably too big.
# rnnlm/train_rnnlm.sh: best iteration (out of 20) was 5, linking it to final iteration.
# Train objf: -99.92 -5.64 -5.11 -4.80 -4.57 -4.36 -4.17 -3.99 -3.82 -3.67 -3.53 -3.40 -3.28 -3.17 -3.07 -2.98 -2.89 -2.81 -2.74 -2.60
# Dev objf:   -9.21 -5.94 -5.32 -5.06 -4.98 -4.97 -4.99 -5.07 -5.15 -5.28 -5.41 -5.52 -5.67 -5.80 -5.92 -6.05 -6.17 -6.30 -6.43 -6.55



set -e
embedding_dim=400
dir=exp/rnnlm_tdnn_a
stage=0
train_stage=0

. utils/parse_options.sh   # parse options-- mainly for the --stage parameter.

for f in data/vocab/words.txt data/text/ptb.txt; do
  [ ! -f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $dir/config
  cp data/vocab/words.txt $dir/config/

  cat > $dir/config/data_weights.txt <<EOF
ptb   1   1.0
EOF

  echo "<unk>" >$dir/config/oov.txt  # not really necessary as OOVs were previously
                              # converted to <unk>, but doesn't hurt.


  # we need the unigram probs to get the unigram features.
  rnnlm/get_unigram_probs.py --vocab-file=data/vocab/words.txt \
                             --data-weights-file=$dir/config/data_weights.txt \
                             data/text >$dir/config/unigram_probs.txt


  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<unk>' \
                           data/vocab/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=400 input=Append(0, IfDefined(-1))
relu-renorm-layer name=tdnn2 dim=400 input=Append(0, IfDefined(-1))
relu-renorm-layer name=tdnn3 dim=400 input=Append(0, IfDefined(-2))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF

  rnnlm/validate_config_dir.sh data/text $dir/config
fi


if [ $stage -le 1 ]; then
  rnnlm/prepare_rnnlm_dir.sh data/text $dir/config $dir
fi

if [ $stage -le 2 ]; then
  rnnlm/train_rnnlm.sh --num-epochs 40 --cmd "queue.pl" $dir
fi



# valid objf.
# grep 'Overall objf' exp/rnnlm_a/log/train.{?,??}.1.log | awk '{printf("%s ", $NF)} END{print "";}'
# -7.198 -5.653 -5.133 -4.82 -4.572 -4.349 -4.137 -3.929 -3.727 -3.531 -3.344 -3.167 -3.001 -2.846 -2.701 -2.568 -2.445 -2.331 -2.226 -2.205

# grep 'Overall objf' exp/rnnlm_a/log/compute_prob.?.log   | awk '{printf("%s ", $NF)} END{print "";}'
# -9.21 -6.185 -5.369 -5.126 -4.993 -4.967 -4.998 -5.092 -5.208 -5.356
