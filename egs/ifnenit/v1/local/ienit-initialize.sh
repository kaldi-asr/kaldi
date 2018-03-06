#!/bin/bash
source cmd.sh
source path.sh

export LC_ALL=en_US.utf8

# Fetch transcriptions, invert images
mkdir -p data/tru


folders="set_a set_b set_c"
touch tmp.flist
for set in 'train' 'test'
do
  rm tmp.flist
  for folder in $folders
  do
    echo "$folder"
    cp /export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/$folder/tru/*.tru data/tru
    ls /export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/$folder/tif/*.tif >> tmp.flist
  done
  cat tmp.flist | xargs -n 1 -IBLA basename BLA '.tif' > tmp.uttids
  cat tmp.flist | sed 's/\/tif\/\([a-z0-9_]\+\).tif$/\/tru\/\1.tru/' | xargs egrep '^LBL:' | iconv -f 'cp1256' -t 'UTF-8' | python3 local/remove_diacritics.py | cut -d':' -f5- | cut -d';' -f1 | paste -d' ' tmp.uttids - > data/text.$set
  folders="set_d"
done
rm tmp.flist tmp.uttids

