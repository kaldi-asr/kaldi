#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <WAV-DIR> <XML-DIR>"
  echo " e.g.: $0 /export/a15/vmanoha1/MGB/eval /export/a15/vmanoha1/MGB/eval_xml_2016_05_29/bw"
  exit 1
fi

wavDir=$1
xmldir=$2

evalDir=data/eval

if [ -f ${evalDir}.uem/wav.scp ]; then
  mkdir -p ${evalDir}.uem/.backup
  mv ${evalDir}.uem/{wav.scp,feats.scp,utt2spk,spk2utt,segments,text} ${evalDir}.uem/.backup
fi

mkdir -p ${evalDir}.uem
for x in $wavDir/*.wav; do
  basename=`basename $x .wav`
  [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
  $XMLSTARLET/xml sel -t -m '//segments[@annotation_id="transcript_manual"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename ${evalDir}.uem
  echo $basename $wavDir/$basename.wav >> ${evalDir}.uem/wav.scp
done

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" 0"}' ${evalDir}.uem/wav.scp > ${evalDir}.uem/reco2file_and_channel

utils/data/convert_data_dir_to_whole.sh ${evalDir}.uem ${evalDir}
