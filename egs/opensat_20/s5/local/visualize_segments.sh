#!/bin/bash
# Apache 2.0

# This script visualize RTTM file with matplotlib
# "rttm: <type> <file-id> <channel-id> <begin-time> <duration> <NA> <NA> <speaker> <conf>"

# get rttm for the oracle data
steps/segmentation/convert_utt2spk_and_segments_to_rttm.py data/safe_t_dev1/utt2spk data/safe_t_dev1/segments data/safe_t_dev1/rttm

# get rttm for the re-segmented data
steps/segmentation/convert_utt2spk_and_segments_to_rttm.py data/safe_t_dev1_segmented_reseg/utt2spk data/safe_t_dev1_segmented_reseg_hires/segments data/safe_t_dev1_segmented_reseg_hires/rttm

#combine rttm from the oracle and resegmented data
cat data/safe_t_dev1/rttm data/safe_t_dev1_segmented_reseg_hires/rttm > data/safe_t_dev1/combine_rttm

# call visualize segments script
local/visualize_segments.py
