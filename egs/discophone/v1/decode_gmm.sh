#!/bin/bash
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

stage=0
decode_nj=20

langs_config="" # conf/experiments/all-ipa.conf
if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
else
  # BABEL TRAIN:
  # Amharic - 307
  # Bengali - 103
  # Cantonese - 101
  # Javanese - 402
  # Vietnamese - 107
  # Zulu - 206
  # BABEL TEST:
  # Georgian - 404
  # Lao - 203
  babel_langs="307 103 101 402 107 206 404 203"
  babel_recog="${babel_langs}"
  gp_langs="Czech French Mandarin Spanish Thai"
  gp_recog="${gp_langs}"
fi

. cmd.sh
. utils/parse_options.sh
. path.sh

local/install_shorten.sh

declare -A recog_to_train
train_set=""
dev_set=""
for l in ${babel_langs}; do
  recog_to_train["eval_${l}"]="$l/data/train_${l}"
  train_set="$l/data/train_${l} ${train_set}"
  dev_set="$l/data/dev_${l} ${dev_set}"
done
for l in ${gp_langs}; do
  recog_to_train["eval_${l}"]="GlobalPhone/gp_${l}_train"
  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
train_set=${train_set%% }
dev_set=${dev_set%% }

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

echo "Training data directories: ${train_set[*]}"
echo "Dev data directories: ${dev_set[*]}"
echo "Eval data directories: ${recog_set[*]}"

function langname() {
  # Utility
  echo "$(basename "$1")"
}

# By default use the multilingual phone LM
lmdir=data/lang_combined_test

if ((stage <= 0)); then
  # Feature extraction
  for data_dir in ${recog_set}; do
    (
      lang_name=$(langname $data_dir)
      steps/make_mfcc.sh \
        --cmd "$train_cmd" \
        --nj 16 \
        --write_utt2num_frames true \
        "data/$data_dir" \
        "exp/make_mfcc/$data_dir" \
        mfcc
      utils/fix_data_dir.sh data/$data_dir
      steps/compute_cmvn_stats.sh data/$data_dir exp/make_mfcc/$lang_name mfcc/$lang_name
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 7)); then
  # Mono decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/mono
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 8)); then
  # Tri1 decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/tri1
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 9)); then
  # Tri2 decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/tri2
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 10)); then
  # Tri3 decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/tri3
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 11)); then
  # Tri4 decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/tri4
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi

if ((stage <= 12)); then
  # Tri5 decoding
  for recog_data_dir in ${recog_set}; do
    (
      # `data_dir` points to the train set directory
      data_dir=${recog_to_train[$recog_data_dir]}
      lang_name=$(langname $data_dir)
      expdir=exp/gmm/$lang_name/tri5
      utils/mkgraph.sh $lmdir $expdir $expdir/graph
      steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
        $expdir/graph data/$recog_data_dir $expdir/decode
    ) &
    sleep 2
  done
  wait
fi
