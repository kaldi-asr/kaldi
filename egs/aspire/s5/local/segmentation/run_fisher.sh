#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

local/segmentation/prepare_fisher_data.sh

local/segmentation/tuning/train_lstm_sad_music_snr_fisher_1k.sh
