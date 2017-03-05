#!/bin/bash

###########################################################################################
# This script was copied from egs/librispeech/s5/local/g2p/train_g2p.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
# - Removed CMUDict download/cleaning steps
# - Changed to use data/local/dict_nosp/lexicon.txt instead
# - Renumbered stages
###########################################################################################

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# Trains Sequitur G2P models on CMUdict

# can be used to skip some of the initial steps
stage=1

. utils/parse_options.sh || exit 1
. ./path.sh || exit 1

if [ $# -ne "2" ]; then
  echo "Usage: $0 <train-lexicon> <g2p-dir>"
  exit 1
fi

train_lex=$1
g2p_dir=$2

mkdir -p $g2p_dir

model_1=$g2p_dir/model-1
model_2=$g2p_dir/model-2
model_3=$g2p_dir/model-3
model_4=$g2p_dir/model-4
model_5=$g2p_dir/model-5

if [ -f $model_5 ]; then
  echo "$model_5 already exists. Skipping G2P model training..."
  exit 0;
fi

if [ $stage -le 1 ]; then
  echo "Training first-order G2P model (log in '$g2p_dir/model-1.log') ..."
  PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
    --train $train_lex --devel 5% --write-model $model_1 >$g2p_dir/model-1.log 2>&1 || exit 1
fi

if [ $stage -le 2 ]; then
  echo "Training second-order G2P model (log in '$g2p_dir/model-2.log') ..."
  PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
    --model $model_1 --ramp-up --train $train_lex \
    --devel 5% --write-model $model_2 >$g2p_dir/model-2.log \
    >$g2p_dir/model-2.log 2>&1 || exit 1
fi

if [ $stage -le 3 ]; then
  echo "Training third-order G2P model (log in '$g2p_dir/model-3.log') ..."
  PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
    --model $model_2 --ramp-up --train $train_lex \
    --devel 5% --write-model $model_3 \
    >$g2p_dir/model-3.log 2>&1 || exit 1
fi

if [ $stage -le 4 ]; then
  echo "Training fourth-order G2P model (log in '$g2p_dir/model-4.log') ..."
  PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
    --model $model_3 --ramp-up --train $train_lex \
    --devel 5% --write-model $model_4 \
    >$g2p_dir/model-4.log 2>&1 || exit 1
fi

if [ $stage -le 5 ]; then
  echo "Training fifth-order G2P model (log in '$g2p_dir/model-5.log') ..."
  PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
    --model $model_4 --ramp-up --train $train_lex \
    --devel 5% --write-model $model_5 \
    >$g2p_dir/model-5.log 2>&1 || exit 1
fi

echo "G2P training finished OK!"
exit 0
