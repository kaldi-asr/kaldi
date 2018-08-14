#!/bin/bash

set -e
stage=0
language_main=Tamil
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p data/local/splits
mkdir -p data/local/text/cleaned
language_lower=$(echo "$language_main" | tr '[:upper:]' '[:lower:]')
if [ $stage -le 0 ]; then
  for language in  english $language_lower; do
    echo "$0: Processing YOMDLE ${language}"
    mkdir -p data/download/${language}/{truth_csv,truth_line_image}
    image/ocr/yomdle/yomdle2csv.py \
      --inputDir /export/corpora5/slam/YOMDLE/final_${language}/ \
      --outputDir data/download/${language}/truth_csv/ \
      --log data/download/yomdle2csv.${language}.log
    local/create_line_image_from_page_image.py \
      /export/corpora5/slam/YOMDLE/final_${language}/images/ \
      data/download/${language}/truth_csv/ \
      data/download/${language}/truth_line_image/ \
      data/local/yomdle-${language}-train.list \
      --filter
  done
fi

if [ $stage -le 1 ]; then
  echo "$0: Processing slam ${language_main}"
  mkdir -p data/download/${language_main}/{truth_csv,truth_line_image}
  image/ocr/yomdle/gedi2csv_enriched.py \
    --inputDir /export/corpora5/slam/SLAM/${language_main}/transcribed/ \
    --outputDir data/download/${language_main}/truth_csv/ \
    --log data/download/gedi2csv.${language_main}.log
  local/create_line_image_from_page_image.py \
    /export/corpora5/slam/SLAM/${language_main}/transcribed/ \
    data/download/${language_main}/truth_csv/ \
    data/download/${language_main}/truth_line_image/ \
    data/local/yomdle-${language_main}-test.list \
    --ext '.png'
fi

if [ $stage -le 2 ]; then
  echo "$0: Processing slam ${language_main}"
  mkdir -p data/download/${language_main}_boxed/{truth_csv,truth_line_image}
  image/ocr/yomdle/gedi2csv_enriched.py \
    --inputDir /export/corpora5/slam/SLAM/${language_main}/boxed/ \
    --ftype boxed \
    --outputDir data/download/${language_main}_boxed/truth_csv/ \
    --log data/download/gedi2csv.${language_main}_boxed.log
  local/create_line_image_from_page_image.py \
    /export/corpora5/slam/SLAM/${language_main}/boxed/ \
    data/download/${language_main}_boxed/truth_csv/ \
    data/download/${language_main}_boxed/truth_line_image/ \
    data/local/yomdle-${language_main}-train_unsup.list \
    --ext '.png' \
    --filter
fi

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

if [ $stage -le 5 ]; then
  cat data/local/yomdle-${language_lower}-train.list data/local/yomdle-english-train.list > data/local/splits/train.txt
  cp data/local/yomdle-${language_main}-test.list data/local/splits/test.txt
  cp data/local/yomdle-${language_main}-train_unsup.list data/local/splits/train_unsup.txt
fi
