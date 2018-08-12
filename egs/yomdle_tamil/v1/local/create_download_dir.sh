#!/bin/bash

set -e
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "Date: $(date)."
mkdir -p data/local/splits
if [ $stage -le 0 ]; then
  for language in  english tamil; do
    echo "$0: Processing YOMDLE ${language}"
    mkdir -p data/download/${language}/{truth_csv,truth_line_image}
    local/yomdle2csv.py \
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
  for language in Tamil; do
    echo "$0: Processing slam ${language}"
    mkdir -p data/download/${language}/{truth_csv,truth_line_image}
    local/gedi2csv_enriched.py \
      --inputDir /export/corpora5/slam/SLAM/${language}/transcribed/ \
      --outputDir data/download/${language}/truth_csv/ \
      --log data/download/gedi2csv.${language}.log
    local/create_line_image_from_page_image.py \
      /export/corpora5/slam/SLAM/${language}/transcribed/ \
      data/download/${language}/truth_csv/ \
      data/download/${language}/truth_line_image/ \
      data/local/yomdle-${language}-test.list \
      --ext '.png'
  done
fi

if [ $stage -le 2 ]; then
  for language in Tamil; do
    echo "$0: Processing slam ${language}"
    mkdir -p data/download/${language}_boxed/{truth_csv,truth_line_image}
    local/gedi2csv_enriched.py \
      --inputDir /export/corpora5/slam/SLAM/${language}/boxed/ \
      --ftype boxed \
      --outputDir data/download/${language}_boxed/truth_csv/ \
      --log data/download/gedi2csv.${language}.log
    local/create_line_image_from_page_image.py \
      /export/corpora5/slam/SLAM/${language}/boxed/ \
      data/download/${language}_boxed/truth_csv/ \
      data/download/${language}_boxed/truth_line_image/ \
      data/local/yomdle-${language}-train_unsup.list \
      --ext '.png' \
      --filter
  done
fi
echo "Date: $(date)."
