#!/usr/bin/env bash
# Copyright 2021  Seasalt AI, Inc (Author: Guoguo Chen)

# This script prepares the data folders for the GigaSpeech dataset.

set -e
set -o pipefail

stage=0
train_subset=XL

. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <gigaspeech-root> <data-dir>"
  echo " e.g.: $0 ~/GigaSpeech_data/dict/g2p/g2p.model.4 data/local/dict"
  echo "Options:"
  echo "  --cmd '<command>'    # script to launch jobs with, default: run.pl"
  echo "  --nj <nj>            # number of jobs to run, default: 4."
  exit 1
fi

gigaspeech_root=`realpath $1`
data_dir=`realpath $2`

gigaspeech_repo=GigaSpeech_repo

echo "$0: Cloning the GigaSpeech repo."
[ -d "$gigaspeech_repo" ] && (rm -rf $gigaspeech_repo || exit 1)
git clone \
  https://github.com/SpeechColab/GigaSpeech.git $gigaspeech_repo || exit 1;
pushd $gigaspeech_repo
git checkout fix_kaldi || exit 1;
popd

if [ $stage -le 0 ]; then
  echo "======GigaSpeech Download START | current time : `date +%Y-%m-%d-%T`==="
  pushd $gigaspeech_repo
  utils/gigaspeech_download.sh $gigaspeech_root || exit 1
  popd
  echo "======GigaSpeech Download END | current time : `date +%Y-%m-%d-%T`====="
fi

if [ $stage -le 1 ]; then
  echo "======GigaSpeech Preparation START | current time : `date +%Y-%m-%d-%T`"
  # Prepare GigaSpeech data
  pushd $gigaspeech_repo
  toolkits/kaldi/gigaspeech_data_prep.sh \
    --train-subset $train_subset $gigaspeech_root $data_dir || exit 1
  popd
  echo "======GigaSpeech Preparation END | current time : `date +%Y-%m-%d-%T`=="
fi


echo "$0: Cleaning up the GigaSpeech repo."
rm -rf $gigaspeech_repo || exit 1

echo "$0: Done"
