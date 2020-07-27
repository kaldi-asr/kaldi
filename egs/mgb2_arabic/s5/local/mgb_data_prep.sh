#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU
#               2016-2019  Vimal Manohar
#               2019 Dongji Gao

if [ $# -ne 3 ]; then
  echo "Usage: $0 <DB-dir> <mer-sel> <process-xml>"
  exit 1;
fi

db_dir=$1
mer=$2

train_dir=data/train_mer$mer
dev_dir=data/dev

for x in $train_dir $dev_dir; do
  mkdir -p $x
  if [ -f ${x}/wav.scp ]; then
    mkdir -p ${x}/.backup
    mv $x/{wav.scp,feats.scp,utt2spk,spk2utt,segments,text} ${x}/.backup
  fi
done

find $db_dir/train/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $train_dir/wav_list

#Creating the train program lists
head -500 $train_dir/wav_list > $train_dir/wav_list.short

set -e -o pipefail

xmldir=$db_dir/train/xml/bw
if [ $process_xml == "python" ]; then
  echo "using python to process xml file"
  # check if bs4 and lxml are installin in python
  local/check_tools.sh
  # process xml file using python
  cat $train_dir/wav_list | while read basename; do
    [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
    local/process_xml.py $xmldir/$basename.xml - | local/add_to_datadir.py $basename $train_dir $mer
  done
elif [ $process_xml == 'xml' ]; then
  # check if xml binary exsits
  if command -v xml >/dev/null 2>/dev/null; then
    echo "using xml"
    cat $train_dir/wav_list | while read basename; do
      [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
      xml sel -t -m '//segments[@annotation_id="transcript_align"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename $train_dir $mer
      echo $basename $wavDir/$basename.wav >> $train_dir/wav.scp
    done
  else
    echo "xml not found, you may use python by '--process-xml python'"
    exit 1;
  fi
else
  # invalid option
  echo "$0: invalid option for --process-xml, choose from 'xml' or 'python'"
  exit 1;
fi

for x in text segments; do
  cp $db_dir/dev/${x}.all $dev_dir/${x}
done

find $db_dir/dev/wav -type f -name "*.wav" | \
  awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
  $dev_dir/wav_list

for x in $(cat $dev_dir/wav_list); do 
  echo $x $db_dir/dev/wav/$x.wav >> $dev_dir/wav.scp
done

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" 1"}' $dev_dir/wav.scp > $dev_dir/reco2file_and_channel

# Creating utt2spk for dev from segments
if [ ! -f $dev_dir/utt2spk ]; then
  cut -d ' ' -f1 $dev_dir/segments > $dev_dir/utt_id
  cut -d '_' -f1-2 $dev_dir/utt_id | paste -d ' ' $dev_dir/utt_id - > $dev_dir/utt2spk
fi

for list in overlap non_overlap; do
  rm -rf ${dev_dir}_$list || true
  cp -r $dev_dir ${dev_dir}_$list
  for x in segments text utt2spk; do
    utils/filter_scp.pl $db_dir/dev/${list}_speech $dev_dir/$x > ${dev_dir}_$list/${x}
  done
done

for dir in $train_dir $dev_dir ${dev_dir}_overlap ${dev_dir}_non_overlap; do
  utils/fix_data_dir.sh $dir
  utils/validate_data_dir.sh --no-feats $dir
done

mkdir -p ${train_dir}_subset500
utils/filter_scp.pl $train_dir/wav_list.short ${train_dir}/wav.scp > \
  ${train_dir}_subset500/wav.scp
cp ${train_dir}/{utt2spk,segments,spk2utt} ${train_dir}_subset500
utils/fix_data_dir.sh ${train_dir}_subset500

echo "Training and Test data preparation succeeded"
