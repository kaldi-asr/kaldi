#!/usr/bin/env bash

# To be run from the directory egs/ptb/s5.
# This is to be done after local/prepare_rnnlm_data.sh.

# some of the output follows.

# rnnlm/train_rnnlm.sh: best iteration (out of 8) was 3, linking it to final iteration.
# Train objf: -92.38 -5.01 -4.70 -4.48 -4.30 -4.14 -4.01 -3.81
# Dev objf:   -9.21 -5.29 -5.05 -4.96 -4.97 -5.02 -5.09 -5.16


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
