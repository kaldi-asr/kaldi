#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script shows how to train and test neural-network 
# speech activity detection using only Fisher data.

if [ ! -d RIRS_NOISES/ ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

local/segmentation/prepare_fisher_data.sh

local/segmentation/tuning/train_lstm_sad_music_snr_fisher_1k.sh

# Assuming frame-subsampling-factor of 3. 
# So that 10 states duration corresponding to 30 frames.
# The format is <class-label> <init-probability> <min-num-states> <list-of-pairs>
# where <list-of-pairs> contains space separated entries of 
# <destination-class:transition-probability>. 
# A destination-class of -1 is used to represent FST final-probability.
cat <<EOF > data/lang_sad/classes_info.txt
1 0.9 0.9 10 2:0.099 -1:0.001
2 0.1 0.9 10 1:0.099 -1:0.001
EOF 

steps/segmentation/do_segmentation_data_dir_simple.sh --nj 30 \
  --convert-data-dir-to-whole true \
  --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp \
  --do-downsampling false \
  --extra-left-context 50 \
  --output-name output-speech --frame-subsampling-factor 3 \
  data/dev_aspire exp/nnet3_lstm_sad_music_snr_fisher/nnet_lstm_1k \
  data/lang_sad mfcc_hires_bp data/dev_aspire_lstm_1k
# Outputs data/dev_aspire_lstm_1k_seg

dir=exp/nnet3/tdnn
local/prep_test_aspire_segmentation.sh --decode-num-jobs 30 --affix "v7" \
   --sub-speaker-frames 6000 --max-count 75 --ivector-scale 0.75  \
   --pass2-decode-opts "--min-active 1000" \
   dev_aspire data/dev_aspire_lstm_1k_seg \
   data/lang exp/tri5a/graph_pp $dir
