#!/usr/bin/env bash
# Copyright 2018 Chun-Chieh Chang

# The original format of the dataset given is GEDI and page images.
# This script is written to create line images from page images.
# It also creates csv files from the GEDI files.

database_train=/export/corpora5/handwriting_ocr/CASIA_HWDB/Offline/
database_competition=/export/corpora5/handwriting_ocr/CASIA_HWDB/Offline/
cangjie_url=https://raw.githubusercontent.com/wanleung/libcangjie/master/tables/cj5-cc.txt
download_dir=download

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1; 

mkdir -p ${download_dir}/{Train,Test}
for task in 0 1 2; do
    for datasplit in Train Test; do
        unzip -q -d ${download_dir}/${datasplit} ${database_train}/CASIA-HWDB2.${task}/${datasplit}_Dgr.zip
    done
done

unzip -q -d ${download_dir}/Competition ${database_competition}/competition-dgr.zip 

echo "Downloading table for CangJie."
wget -P $download_dir/ $cangjie_url || exit 1;
sed -ie '1,8d' $download_dir/cj5-cc.txt
