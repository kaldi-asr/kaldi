#!/bin/bash
. ./cmd.sh
. ./path.sh

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

data_dir="/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e"

export LC_ALL="en_US.utf8"

# Fetch transcriptions for Test and Train.
# If you want try different sets for test and train
# you should change this file and the "local/process_data.py" script.

mkdir -p data/tru

folders="set_a set_b set_c"
touch tmp.flist
for set in 'train' 'test'
do
  rm tmp.flist
  for folder in $folders
  do
    echo "$folder"
    cp $data_dir/data/$folder/tru/*.tru data/tru
    ls $data_dir/data/$folder/png/*.png >> tmp.flist
  done
  cat tmp.flist | xargs -n 1 -IBLA basename BLA '.png' > tmp.uttids
  cat tmp.flist | sed 's/\/png\/\([a-z0-9_]\+\).png$/\/tru\/\1.tru/' | xargs egrep '^LBL:' | iconv -f 'cp1256' -t 'UTF-8' | python3 local/remove_diacritics.py | cut -d':' -f5- | cut -d';' -f1 | paste -d' ' tmp.uttids - > data/text.$set
  folders="set_d"
done
rm tmp.flist tmp.uttids

export LC_ALL=C