#!/bin/bash

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

database_dir= # directory of the dataset
train_sets= # sets for training
test_sets=  # sets for testing

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# Fetch transcriptions for Test and Train.
# If you want try different sets for test and train
# you should change this file and the "local/process_data.py" script.

mkdir -p data/tru

folders=$train_sets
touch tmp.flist
for set in 'train' 'test'
do
  rm tmp.flist
  for folder in $folders
  do
    echo "$folder"
    cp $database_dir/$folder/tru/*.tru data/tru
    ls $database_dir/$folder/png/*.png >> tmp.flist
  done
  cat tmp.flist | xargs -n 1 -IBLA basename BLA '.png' > tmp.uttids
  cat tmp.flist | sed 's/\/png\/\([a-z0-9_]\+\).png$/\/tru\/\1.tru/' | xargs egrep '^LBL:' | iconv -f 'cp1256' -t 'UTF-8' | python3 local/remove_diacritics.py | cut -d':' -f5- | cut -d';' -f1 | paste -d' ' tmp.uttids - > data/text.$set
  folders=$test_sets
done
rm tmp.flist tmp.uttids
