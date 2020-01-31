#!/usr/bin/env bash
# Copyright 2018 Chun-Chieh Chang

# The original format of the dataset given is GEDI and page images.
# This script is written to create line images from page images.
# It also creates csv files from the GEDI files.

database_slam=/export/corpora5/slam/SLAM/Farsi/transcribed
database_yomdle=/export/corpora5/slam/YOMDLE/final_farsi
cangjie_url=https://raw.githubusercontent.com/wanleung/libcangjie/master/tables/cj5-cc.txt
download_dir=download
slam_dir=$download_dir/slam_farsi
yomdle_dir=$download_dir/yomdle_farsi

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1; 

echo "$0: Processing SLAM ${language}"
echo "Date: $(date)."
mkdir -p ${slam_dir}/{truth_csv,truth_csv_raw,truth_line_image}
local/gedi2csv.py \
    --inputDir ${database_slam} \
    --outputDir ${slam_dir}/truth_csv_raw \
    --log ${slam_dir}/GEDI2CSV_enriched.log
local/create_line_image_from_page_image.py \
    ${database_slam} \
    ${slam_dir}/truth_csv_raw \
    ${slam_dir}

echo "$0: Processing YOMDLE ${language}"
echo "Date: $(date)."
mkdir -p ${yomdle_dir}/{truth_csv,truth_csv_raw,truth_line_image}
local/yomdle2csv.py \
    --inputDir ${database_yomdle} \
    --outputDir ${yomdle_dir}/truth_csv_raw/ \
    --log ${yomdle_dir}/YOMDLE2CSV.log
local/create_line_image_from_page_image.py \
    --im-format "jpg" \
    ${database_yomdle}/images \
    ${yomdle_dir}/truth_csv_raw \
    ${yomdle_dir}

echo "Downloading table for CangJie."
wget -P $download_dir/ $cangjie_url || exit 1;
perl -n -i -e 'print if $. > 8' $download_dir/cj5-cc.txt
