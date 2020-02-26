#!/bin/bash

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

. ./cmd.sh
. ./path.sh

data=/home/fangjun/data/aishell
data_url=www.openslr.org/resources/33

nj=30

stage=8

if [[ $stage -le 0 ]]; then
  local/download_and_untar.sh $data $data_url data_aishell || exit 1
  local/download_and_untar.sh $data $data_url resource_aishell || exit 1
fi

if [[ $stage -le 1 ]]; then
  local/aishell_prepare_dict.sh $data/resource_aishell || exit 1
  # generated in data/local/dict
fi

if [[ $stage -le 2 ]]; then
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript || exit 1
  # generated in data/{train,test,dev}/{spk2utt text utt2spk wav.scp}
fi

if [[ $stage -le 3 ]]; then
  local/aishell_train_lms.sh || exit 1
fi

if [[ $stage -le 4 ]]; then
  echo "$0: generating TLG.fst"
  ./local/generate_tlg.sh \
    data/local/dict/lexicon.txt \
    data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/lang
fi

if [[ $stage -le 5 ]]; then
  echo "$0: generating fbank features (40-dim)"

  for x in train dev; do
    utils/data/perturb_data_dir_speed_3way.sh data/$x data/${x}_sp
  done

  for x in train_sp dev_sp test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data/$x || exit 1
    steps/compute_cmvn_stats.sh data/$x || exit 1
    utils/fix_data_dir.sh data/$x || exit 1
  done
fi

if [[ $stage -le 6 ]]; then
  echo "$0: convert text to labels"
  for x in train_sp dev_sp test; do
    ./local/convert_text_to_labels.sh data/$x data/lang
  done
fi

# n=1024
# # n=
# if [[ $stage -le 7 ]]; then
#   if true; then
#     utils/subset_data_dir.sh data/train_sp $n data/train_sp$n || exit 1
#     utils/subset_data_dir.sh data/dev_sp $n data/dev_sp$n || exit 1
#   else
#     utils/subset_data_dir.sh --first data/train_sp $n data/train_sp$n || exit 1
#     utils/subset_data_dir.sh --first data/dev_sp $n data/dev_sp$n || exit 1
#   fi
#
#   for x in train_sp dev_sp; do
#     ./local/convert_text_to_labels.sh data/${x}$n data/lang
#   done
# fi

if [[ $stage -le 8 ]]; then
  ./local/run_ctc.sh \
    --train-data-dir data/train_sp$n \
    --dev-data-dir data/dev_sp$n \
    --test-data-dir data/test  \
    --lang-dir data/lang \
    --nj $nj
fi
