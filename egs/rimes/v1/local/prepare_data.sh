#!/bin/bash

# This script creates traing and validations splits, downloads text corpus for language modeling,
#  prepares the training, validation and test data for rimes dataset 
# (i.e text, images.scp, utt2spk and spk2utt). It calls process_data.py.

#  Eg. local/prepare_data.sh
#  Eg. text file: writer000150_train2011-150_000001 J'ai perdu mon emploi depuis 3 mois et je me
#      utt2spk file: writer000150_train2011-150_000001 writer000150
#      images.scp file: writer000150_train2011-150_000001 data/local/rimes_data/line_image/train/train2011-150_000001.png

stage=0
download_dir=data/local/rimes_data
data_dir=data/local/rimes_data
page_image=$data_dir/page_image
xml=$data_dir/xml
train_img_url="http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:training_2011.tar";
train_xml_url="http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:training_2011.xml";
test_xml_url="http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:eval_2011_annotated.xml";
test_img_url="http://www.a2ialab.com/lib/exe/fetch.php?media=rimes_database:data:icdar2011:line:eval_2011.tar";
text_url="http://opus.nlpl.eu/download.php?f=OfisPublik.tar.gz"
use_extra_corpus_text=true
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p data/{train,test,val}

if [ -d $page_image ]; then
  echo "$0: Not downloading data as it is already there."
else
  mkdir -p $data_dir/{page_image,xml,line_image}/{train_total,test,val,train}
  tar -xf $download_dir/training_2011.tar -C $page_image/train_total || exit 1;
  tar -xf $download_dir/eval_2011.tar -C $page_image/test || exit 1;
  cp -r $download_dir/training_2011.xml $xml/train_total/rimes_2011.xml
  cp -r $download_dir/eval_2011_annotated.xml $xml/test/rimes_2011.xml
  echo "$0: Done downloading and extracting data"

  #First 150 training page images are used for validation  
  cat $xml/train_total/rimes_2011.xml | head -n451  > $xml/val/rimes_2011.xml
  cat $xml/train_total/rimes_2011.xml | tail -1  >> $xml/val/rimes_2011.xml
  cp -r $page_image/train_total/* $page_image/train

  #Remaining training page images are used for training
  cat $xml/train_total/rimes_2011.xml | head -1  > $xml/train/rimes_2011.xml
  cat $xml/train_total/rimes_2011.xml | tail -n+452  >> $xml/train/rimes_2011.xml
  cp -r $page_image/train_total/* $page_image/val
fi

if $use_extra_corpus_text; then
  # using freely available french text corpus for language modeling
  mkdir -p data/local/text_data
  wget -P data/local/text_data $text_url || exit 1;
  tar -xf data/local/text_data/download.php?f=OfisPublik.tar.gz -C data/local/text_data || exit 1;
  zcat data/local/text_data/OfisPublik/raw/fr/*.gz > data/local/text_data/fr_text
fi

if [ $stage -le 0 ]; then
  echo "$0: Processing train, val and test data... $(date)."
  local/process_data.py $data_dir train --augment true || exit 1
  local/process_data.py $data_dir val || exit 1
  local/process_data.py $data_dir  test || exit 1
  for dataset in test train val; do
    echo "$0: Fixing data directory for dataset: $dataset $(date)."
    image/fix_data_dir.sh data/$dataset
  done
fi
