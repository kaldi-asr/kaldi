#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU
#               2016-2019  Vimal Manohar

if [ $# -ne 3 ]; then
  echo "Usage: $0 <wav-dir> <xml-dir> <mer-sel>"
  exit 1;
fi

wavDir=$1
xmldir=$2
mer=$3

#wavDir=$WAV_DIR;xmldir=$XML_DIR;mer=80
trainDir=data/train_mer$mer
devDir=data/dev

rm -r $trainDir || true
rm -r $devDir || true
mkdir -p $trainDir
mkdir -p $devDir

dev=dev$$

set -e -o pipefail

cut -d '/' -f2 local/train | while read basename; do     
    [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
    $XMLSTARLET/xml sel -t -m '//segments[@annotation_id="transcript_align"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename $trainDir $mer
    echo $basename $wavDir/$basename.wav >> $trainDir/wav.scp
done 

cut -d '/' -f2 local/dev | while read basename; do
    [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
    $XMLSTARLET/xml sel -t -m '//segments[@annotation_id="transcript_manual"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename $devDir
    echo $basename $wavDir/$basename.wav >> $devDir/wav.scp
done

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" 0"}' $devDir/wav.scp > $devDir/reco2file_and_channel

#stm reference file for scoring
cat dev$$ | while read basename; do
    [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
    local/xml2stm.py $xmldir/$basename.xml 
done > $devDir/stm

for list in overlap non_overlap; do
  rm -rf ${devDir}_$list || true
  cp -r $devDir ${devDir}_$list
  for x in segments text utt2spk; do
    utils/filter_scp.pl local/${list}_speech.lst $devDir/$x > ${devDir}_$list/$x
  done
done

for dir in $trainDir $devDir ${devDir}_overlap ${devDir}_non_overlap; do
  utils/fix_data_dir.sh $dir
  utils/validate_data_dir.sh --no-feats $dir
done

for dir in ${devDir} ${devDir}_overlap ${devDir}_non_overlap; do
  awk '{print $1 " " $1}' $dir/segments > $dir/spk2utt
  cp $dir/spk2utt $dir/utt2spk
  perl -e '
 ($f1,$f2)= split /\s+/, $ARGV[0];
 open(FNAME, "$f1");
 while (<FNAME>){chomp $_;@arr=split /\s+/,$_;shift @arr;$scal = "@arr";$hashExist{$scal}=1;}close (FNAME);
 open(FTR, "$f2"); while (<FTR>){$line=$_;s/ 0 UNKNOWN / /;@arr=split /\s+/,$_;if (defined $hashExist{"$arr[0] $arr[1] $arr[2]"}) {print "$line";}}close (FTR);
 ' "$dir/segments $dir/stm" > $dir/stm_
 mv $dir/stm_ $dir/stm
done

echo "Training and Test data preparation succeeded"
