#!/usr/bin/env bash

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Hossein Hadian
# Apache 2.0

# This script downloads the IAM handwriting database and prepares the training
# and test data (i.e text, images.scp, utt2spk and spk2utt) by calling process_data.py.
# It also downloads the LOB and Brown text corpora. It downloads the database files
# only if they do not already exist in download directory.

#  Eg. local/prepare_data.sh
#  Eg. text file: 000_a01-000u-00 A MOVE to stop Mr. Gaitskell from
#      utt2spk file: 000_a01-000u-00 000
#      images.scp file: 000_a01-000u-00 data/local/lines/a01/a01-000u/a01-000u-00.png
#      spk2utt file: 000 000_a01-000u-00 000_a01-000u-01 000_a01-000u-02 000_a01-000u-03

stage=0
download_dir=data/download
process_aachen_split=false
wellington_dir=
username=
password=       # username and password for downloading the IAM database
                # if you have not already downloaded the database, please
                # register at http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
                # and provide this script with your username and password.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

if [[ ! -f $download_dir/lines.tgz && -z $username ]]; then
  echo "$0: Warning: Couldn't find lines.tgz in $download_dir. Unless the extracted dataset files"
  echo "exist in your data/local directory this script will fail because the required files"
  echo "can't be downloaded automatically (it needs registration)."
  echo "Please register at http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"
  echo "... and then call this script again with --username <username> --password <password>"
  echo ""
  exit 1
fi

lines=data/local/lines
xml=data/local/xml
ascii=data/local/ascii
bcorpus=data/local/browncorpus
lobcorpus=data/local/lobcorpus
wcorpus=data/local/wellingtoncorpus
data_split_info=data/local/largeWriterIndependentTextLineRecognitionTask
lines_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
xml_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz
data_split_info_url=http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
ascii_url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz
brown_corpus_url=http://www.sls.hawaii.edu/bley-vroman/brown.txt
lob_corpus_url=http://ota.ox.ac.uk/text/0167.zip
wellington_corpus_loc=/export/corpora5/Wellington/WWC/
aachen_split_url=http://www.openslr.org/resources/56/splits.zip
aachen_splits=data/local/aachensplits
mkdir -p $download_dir data/local

# download and extact images and transcription
if [ -d $lines ]; then
  echo "$0: Not downloading lines images as it is already there."
else
  if [ ! -f $download_dir/lines.tgz ]; then
    echo "$0: Trying to download lines images..."
    wget -P $download_dir --user "$username" --password "$password" $lines_url || exit 1;
  fi
  mkdir -p $lines
  tar -xzf $download_dir/lines.tgz -C $lines || exit 1;
  echo "$0: Done downloading and extracting lines images"
fi

if [ -d $xml ]; then
  echo "$0: Not downloading transcriptions as it is already there."
else
  if [ ! -f $download_dir/xml.tgz ]; then
    echo "$0: Trying to download transcriptions..."
    wget -P $download_dir --user "$username" --password "$password" $xml_url || exit 1;
  fi
  mkdir -p $xml
  tar -xzf $download_dir/xml.tgz -C $xml || exit 1;
  echo "$0: Done downloading and extracting transcriptions."
fi

if [ -d $data_split_info ]; then
  echo "$0: Not downloading data split information as it is already there."
else
  if [ ! -f $download_dir/largeWriterIndependentTextLineRecognitionTask.zip ]; then
    echo "$0: Trying to download training and testing data split information..."
    wget -P $download_dir --user "$username" --password "$password" $data_split_info_url || exit 1;
  fi
  mkdir -p $data_split_info
  unzip $download_dir/largeWriterIndependentTextLineRecognitionTask.zip -d $data_split_info || exit 1;
  echo "$0: Done downloading and extracting training and testing data split information"
fi

if [ -d $ascii ]; then
  echo "$0: Not downloading ascii.tgz as it is already there."
else
  if [ ! -f $download_dir/ascii.tgz ]; then
    echo "$0: trying to download ascii.tgz..."
    wget -P $download_dir --user "$username" --password "$password" $ascii_url || exit 1;
  fi
  mkdir -p $ascii
  tar -xzf $download_dir/ascii.tgz -C $ascii || exit 1;
  echo "$0: Done downloading and extracting ascii.tgz"
fi

if [ -d $lobcorpus ]; then
  echo "$0: Not downloading the LOB text corpus as it is already there."
else
  if [ ! -f $lobcorpus/0167.zip ]; then
    echo "$0: Downloading the LOB text corpus ..."
    mkdir -p $lobcorpus
    wget -P $lobcorpus/ $lob_corpus_url || exit 1;
  fi
  unzip $lobcorpus/0167.zip -d $lobcorpus || exit 1;
  echo "$0: Done downloading and extracting LOB corpus"
fi

if [ -d $bcorpus ]; then
  echo "$0: Not downloading the Brown corpus as it is already there."
else
  if [ ! -f $bcorpus/brown.txt ]; then
    mkdir -p $bcorpus
    echo "$0: Downloading the Brown text corpus..."
    wget -P $bcorpus $brown_corpus_url || exit 1;
  fi
  echo "$0: Done downloading the Brown text corpus"
fi

if [ -d $wcorpus ]; then
  echo "$0: Not copying Wellington corpus as it is already there."
elif [ ! -z $wellington_dir ]; then
  mkdir -p $wcorpus
  cp -r $wellington_dir/. $wcorpus

  # Combine Wellington corpora and replace some of their annotations
  cat data/local/wellingtoncorpus/Section{A,B,C,D,E,F,G,H,J,K,L}.txt | \
    cut -d' ' -f3- | sed "s/^[ \t]*//" > data/local/wellingtoncorpus/Wellington_annotated.txt

  cat data/local/wellingtoncorpus/Wellington_annotated.txt | local/remove_wellington_annotations.py > data/local/wellingtoncorpus/Wellington_annotation_removed.txt

  echo "$0: Done copying Wellington corpus"
else
  echo "$0: Wellington Corpus not included because wellington_dir not provided"
fi

if [ -d $aachen_splits ]; then
  echo "$0: Not downloading the Aachen splits as it is already there."
else
  if [ ! -f $aachen_splits/splits.zip ]; then
    echo "$0: Downloading Aachen splits ..."
    mkdir -p $aachen_splits
    wget -P $aachen_splits/ $aachen_split_url || exit 1;
  fi
  unzip $aachen_splits/splits.zip -d $aachen_splits || exit 1;
  echo "$0: Done downloading and extracting Aachen splits"
fi


mkdir -p data/{train,test,val}
file_name=largeWriterIndependentTextLineRecognitionTask

train_old="data/local/$file_name/trainset.txt"
test_old="data/local/$file_name/testset.txt"
val1_old="data/local/$file_name/validationset1.txt"
val2_old="data/local/$file_name/validationset2.txt"

train_new="data/local/train.uttlist"
test_new="data/local/test.uttlist"
val_new="data/local/validation.uttlist"

cat $train_old > $train_new
cat $test_old > $test_new
cat $val1_old $val2_old > $val_new

if $process_aachen_split; then
    local/process_aachen_splits.py data/local $aachen_splits/splits data/train --dataset train || exit 1
    local/process_aachen_splits.py data/local $aachen_splits/splits data/test --dataset test || exit 1
    local/process_aachen_splits.py data/local $aachen_splits/splits data/val --dataset validation || exit 1
else
    local/process_data.py data/local data/train --dataset train || exit 1
    local/process_data.py data/local data/test --dataset test || exit 1
    local/process_data.py data/local data/val --dataset validation || exit 1
fi

image/fix_data_dir.sh data/train
image/fix_data_dir.sh data/test
image/fix_data_dir.sh data/val
