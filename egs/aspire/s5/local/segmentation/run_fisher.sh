#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

local/segmentation/prepare_fisher_data.sh

utils/combine_data.sh --extra-files "speech_feat.scp deriv_weights.scp deriv_weights_manual_seg.scp music_labels.scp" \
  data/fisher_train_100k_whole_all_corrupted_sp_hires_bp \
  data/fisher_train_100k_whole_corrupted_sp_hires_bp \
  data/fisher_train_100k_whole_music_corrupted_sp_hires_bp

local/segmentation/train_stats_sad_music.sh \
  --train-data-dir data/fisher_train_100k_whole_all_corrupted_sp_hires_bp \
  --speech-feat-scp data/fisher_train_100k_whole_corrupted_sp_hires_bp/speech_feat.scp \
  --deriv-weights-scp data/fisher_train_100k_whole_corrupted_sp_hires_bp/deriv_weights.scp \
  --music-labels-scp data/fisher_train-100k_whole_music_corrupted_sp_hires_bp/music_labels.scp \
  --max-param-change 0.2 \
  --num-epochs 2 --affix k \
  --splice-indexes "-3,-2,-1,0,1,2,3 -6,0,mean+count(-99:3:9:99) -9,0,3 0" 

local/segmentation/run_segmentation_ami.sh \
  --nnet-dir exp/nnet3_sad_snr/nnet_tdnn_k_n4
