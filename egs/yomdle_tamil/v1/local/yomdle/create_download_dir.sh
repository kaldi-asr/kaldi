#!/bin/bash

# Copyright      2018  Chun Chieh Chang
#                2018  Ashish Arora
#                2018  Hossein Hadian
# Apache 2.0

# This script assumes that the SLAM and Yomdle OCR database is stored in slam_dir and
# yomdle_dir. It reads the xml files and converts them to csv files. It then with the
# help of csv files, extracts lines images from page images. It can create dataset for
# any yomdle and slam language. Assuming it is creating dataset for Tamil OCR. It
# creates csv files for yomdle English, yomdle Tamil, slam Tamil transcribed and slam
# Tamil boxed. It also creates train, test and train_unsup sets for training and testing.
# Yomdle (English and Tamil) is training set, slam Tamil transcribed is test set, and
# slam Tamil boxed is semi-supervised set.

set -e
stage=0
language_main=Tamil
slam_dir=/export/corpora5/slam/SLAM/
yomdle_dir=/export/corpora5/slam/YOMDLE/

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p data/local/splits
language_lower=$(echo "$language_main" | tr '[:upper:]' '[:lower:]')

echo "$0: extracting line images for english and ${language} for shared model training"
if [ $stage -le 0 ]; then
  for language in  english $language_lower; do
    echo "$0: Processing YOMDLE ${language}"
    mkdir -p data/download/${language}/{truth_csv,truth_line_image}
    local/yomdle/yomdle2csv.py \
      --inputDir $yomdle_dir/final_$language/ \
      --outputDir data/download/${language}/truth_csv/ \
      --log data/download/yomdle2csv.${language}.log
    local/yomdle/create_line_image_from_page_image.py \
      $yomdle_dir/final_$language/images/ \
      data/download/${language}/truth_csv/ \
      data/download/${language}/truth_line_image/ \
      data/local/yomdle-${language}-train.list \
      --filter
  done
fi

echo "$0: extracting line images for slam ${language} for testing"
if [ $stage -le 1 ]; then
  echo "$0: Processing slam ${language_main}"
  mkdir -p data/download/${language_main}/{truth_csv,truth_line_image}
  local/yomdle/gedi2csv_enriched.py \
    --inputDir $slam_dir/${language_main}/transcribed/ \
    --outputDir data/download/${language_main}/truth_csv/ \
    --log data/download/gedi2csv.${language_main}.log
  local/yomdle/create_line_image_from_page_image.py \
    $slam_dir/${language_main}/transcribed/ \
    data/download/${language_main}/truth_csv/ \
    data/download/${language_main}/truth_line_image/ \
    data/local/yomdle-${language_main}-test.list \
    --ext '.png'
fi

echo "$0: extracting line images for semi supervised training for slam ${language}"
if [ $stage -le 2 ]; then
  echo "$0: Processing slam ${language_main}"
  mkdir -p data/download/${language_main}_boxed/{truth_csv,truth_line_image}
  local/yomdle/gedi2csv_enriched.py \
    --inputDir $slam_dir/${language_main}/boxed \
    --ftype boxed \
    --outputDir data/download/${language_main}_boxed/truth_csv/ \
    --log data/download/gedi2csv.${language_main}_boxed.log
  local/yomdle/create_line_image_from_page_image.py \
    $slam_dir/${language_main}/boxed \
    data/download/${language_main}_boxed/truth_csv/ \
    data/download/${language_main}_boxed/truth_line_image/ \
    data/local/yomdle-${language_main}-train_unsup.list \
    --ext '.png' \
    --filter
fi

echo "$0: storing english, given language(transcribed and untranscribed) line images together"
if [ $stage -le 3 ]; then
  cp -r data/download/${language_main}_boxed/truth_line_image/* data/download/$language_lower/truth_line_image/
  cp -r data/download/$language_main/truth_line_image/* data/download/$language_lower/truth_line_image/
  cp -r data/download/english/truth_line_image/* data/download/$language_lower/truth_line_image/
  cp -r data/download/${language_main}_boxed/truth_csv/* data/download/$language_lower/truth_csv/
  cp -r data/download/$language_main/truth_csv/* data/download/$language_lower/truth_csv/
  cp -r data/download/english/truth_csv/* data/download/$language_lower/truth_csv/
fi


if [ $stage -le 4 ]; then
  mv data/download/$language_lower/truth_line_image/ data/download/
  mv data/download/$language_lower/truth_csv/ data/download/
fi

echo "$0: storing train, test and train unsupervised splits"
if [ $stage -le 5 ]; then
  cat data/local/yomdle-${language_lower}-train.list data/local/yomdle-english-train.list > data/local/splits/train.txt
  cp data/local/yomdle-${language_main}-test.list data/local/splits/test.txt
  cp data/local/yomdle-${language_main}-train_unsup.list data/local/splits/train_unsup.txt
fi
