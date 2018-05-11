#!/bin/bash

# Note: this works only on pre-downloaded data on the CLSP servers
data=/export/a05/dgalvez/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=4
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data
#Stage1: run run.sh from scratch to generate a chain model.
if [ $stage -le 0 ]; then
  run.sh
fi

#Stage2: prepare a noise dir(maybe a speicial noise dataset). In mini_librispeech,
#we just use trainset directly.
if [ $stage -le 1 ]; then
  cp -r data/train_clean_5 data/noise
  #for the noise dir, we prepare a file utt2dur_fix. Each line is "utt_id dur-0.2"
  #This file is used in "fvector-chunk.cc". It will be store into a vector in binary code.
  #For each target chunk, we randomly select two utt_id form vector, and the 
  #corresponding start point.
  utils/data/get_utt2dur.sh data/noise  # wav-to-duration
  cat data/noise/utt2dur | awk '{print $1,$2-0.2}' > data/noise/utt2dur_fix
fi

if [ $stage -le 2 ]; then
#generate fvector egs and train model.
local/fvector/run_fvector.sh --data data/train_clean_5 --noise-data data/noise \
  --egs-dir exp/fvector/egs --fvector-dir exp/fvector
fi

if [ $stage -le 3 ]; then
  for part in dev_clean_2_hires train_clean_5_sp_hires; do
    if [ -e data/${part}_mfcc ]; then
      if [ -e data/${part} ]; then
        rm -rf data/${part}
      fi
      mv data/${part}_mfcc data/${part}
    fi
    
    mv data/${part} data/${part}_mfcc
    cp -r data/${part}_mfcc data/${part}
    for f in $(ls data/${part}); do
      if [ $f != "spk2gender" -a $f != "spk2utt" -a $f != "text" -a $f != "utt2spk" -a $f != "wav.scp" ]; then
        rm -rf data/$part/$f
      fi
    done
    steps/nnet3/fvector/make_fvector_feature.sh --cmd "$train_cmd" --nj 10 \
      data/${part} exp/fvector exp/make_fvector/train fvector_feature
  done
fi

if [ $stage -le 4 ]; then
  local/fvector/run_tdnn.sh --stage 14 --train-stage 9
fi
