#!/bin/bash

set -e
stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "Date: $(date)."
mkdir -p data/local/splits
mkdir -p data/local/text/cleaned
if [ $stage -le 0 ]; then
  for language in  english tamil; do
    echo "$0: Processing YOMDLE ${language}"
    mkdir -p data/download/${language}/{truth_csv,truth_line_image}
    image/ocr/yomdle2csv.py \
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
    image/ocr/gedi2csv_enriched.py \
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
    image/ocr/gedi2csv_enriched.py \
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

if [ $stage -le 3 ]; then
  cp -r data/download/Tamil_boxed/truth_line_image/* data/download/tamil/truth_line_image/
  cp -r data/download/Tamil/truth_line_image/* data/download/tamil/truth_line_image/
  cp -r data/download/english/truth_line_image/* data/download/tamil/truth_line_image/
  cp -r data/download/Tamil_boxed/truth_csv/* data/download/tamil/truth_csv/
  cp -r data/download/Tamil/truth_csv/* data/download/tamil/truth_csv/
  cp -r data/download/english/truth_csv/* data/download/tamil/truth_csv/
fi


if [ $stage -le 4 ]; then
  cat data/local/yomdle-tamil-train.list data/local/yomdle-english-train.list > data/local/splits/yomdle-tamil-train.list
  cp data/local/yomdle-Tamil-test.list data/local/splits/yomdle-tamil-test.list
  cp data/local/yomdle-Tamil-train_unsup.list data/local/splits/yomdle-tamil-train_unsup.list
fi

if [ $stage -le 5 ]; then
  cat /export/corpora5/handwriting_ocr/corpus_data/ta/* > data/local/text/corpus.txt
  head -20000 data/local/text/corpus.txt > data/local/text/val.txt
  tail -n +20000 data/local/text/corpus.txt > data/local/text/ta.txt
fi
echo "Date: $(date)."
