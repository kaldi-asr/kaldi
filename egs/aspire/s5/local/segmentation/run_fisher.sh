#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

if [ ! -d RIRS_NOISES/ ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  unzip rirs_noises.zip
fi

local/segmentation/prepare_fisher_data.sh

local/segmentation/tuning/train_lstm_sad_music_snr_fisher_1k.sh

cat <<EOF > data/lang_sad/classes_info.txt
1 0.9 0.9 30 2:0.09 -1:0.01
2 0.1 0.9 30 1:0.09 -1:0.01
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
