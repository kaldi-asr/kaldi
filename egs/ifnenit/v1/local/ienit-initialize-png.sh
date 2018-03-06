#!/bin/bash
source cmd.sh
source path.sh

export LC_ALL=en_US.utf8

# Fetch transcriptions, invert images
# mkdir -p data/binaries.tmp
# mkdir -p data/tru
# mkdir -p data/normalized

folders="set_e set_f set_s set_a set_b set_c set_d"
touch tmp.flist
# for set in 'train' 'test'
# do
for folder in $folders
do
  echo "$folder"
  # cp /export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/$folder/tru/*.tru data/tru
  ls /export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/$folder/tif/*.tif > tmp.flist

# cat tmp.flist | xargs -n 1 -IBLA basename BLA '.tif' > tmp.uttids
$train_cmd JOB=1 ./log/$set.normalize.log /export/b01/babak/prepocressor-0.2.1/prepocressor -inputFile tmp.flist -outputPath "/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/$folder/png/%base.png" -pipeline 'grayscale' -nThreads 8
# cat tmp.flist | sed 's/\/tif\/\([a-z0-9_]\+\).tif$/\/tru\/\1.tru/' | xargs egrep '^LBL:' | iconv -f 'cp1256' -t 'UTF-8' | python3 scripts/remove_diacritics.py | cut -d':' -f5- | cut -d';' -f1 | paste -d' ' tmp.uttids - > data/text.$set
done
# done
rm tmp.flist tmp.uttids

# python local/scale.py

# mkdir -p data/binaries
# # Save images with QATIP ids
# for img in $(ls data/binaries.tmp/*.png)
# do
  # mv $img data/binaries/$(echo $img | xargs -n 1 -IBLA basename BLA '.png' | scripts/convert-to-qatip-id.sh ienit).png
# done
# rmdir data/binaries.tmp
