#!/bin/bash

# Copyright  2018-2020  Yiming Wang
#            2018-2020  Daniel Povey
# Apache 2.0

# This script loads the Mobvoi dataset.
[ -f ./path.sh ] && . ./path.sh

dl_dir=data/download

mkdir -p $dl_dir

src_path=/export/fs04/a11/hlyu/wakeup_word_corpra/mobvoi

dataset=ticmini2_dataset_20180607.zip
if [ -d $dl_dir/$(basename "$dataset" .zip) ]; then
  echo "Not extracting $(basename "$dataset" .zip) as it is already there."
else
  if [ ! -f $dl_dir/$dataset ]; then
    echo "Downloading $dataset..."
    cat $src_path/ticmini2_dataset_20180607.z01 $src_path/$dataset > $dl_dir/$dataset
  fi
  unzip $dl_dir/$dataset -d $dl_dir
  rm -f $dl_dir/$dataset 2>/dev/null || true
  echo "Done extracting $dataset."
fi

dataset=ticmini2_for_school_20180911.tar.gz
if [ -d $dl_dir/$(basename "$dataset" .tar.gz) ]; then
  echo "Not extracting $(basename "$dataset" .tar.gz) as it is already there."
else
  echo "Extracting $dataset..."
  tar -xvzf $src_path/$dataset -C $dl_dir || exit 1;
  echo "Done extracting $dataset."
fi

dataset=ticmini2_hixiaowen_adult_20180731.7z
if [ -d $dl_dir/$(basename "$dataset" .7z) ]; then
  echo "Not extracting $(basename "$dataset" .7z) as it is already there."
else
  echo "Extracting $dataset..."
  ~/p7zip_16.02/bin/7z x $src_path/$dataset -o$dl_dir|| exit 1;
  echo "Done extracting $dataset."
fi

for dataset in train dev eval; do
  cp $src_path/${dataset}_list $dl_dir/${dataset}_list
done

exit 0
