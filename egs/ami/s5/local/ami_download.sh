#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)

if [ $# -ne 2 ]; then
  echo "Usage: $0 <mic> <ami-dir>"
  echo " where <mic> is either ihm, sdm or mdm and <ami-dir> is download space."
  exit 1;
fi

mic=$1
adir=$2
amiurl=http://groups.inf.ed.ac.uk/ami
annotver=ami_public_manual_1.6.1.zip
wdir=data/local/downloads

if [[ ! "$mic" =~ ^(ihm|sdm|mdm)$ ]]; then
  echo "$0. Wrong <mic> option." 
  exit 1;
fi

mics="1 2 3 4 5 6 7 8"
if [ "$mic" == "sdm" ]; then
  mics=1
fi

mkdir -p $adir
mkdir -p $wdir/log

#download annotations

annot="$adir/$annotver"
if [[ ! -d $adir/annotations || ! -f "$annot" ]]; then
  echo "Downloading annotiations..."
  wget -O $annot $amiurl/AMICorpusAnnotations/$annotver &> $wdir/log/download_ami_annot.log
  mkdir $adir/annotations
  unzip -d $adir/annotations $annot &> /dev/null
fi
[ ! -f "$adir/annotations/AMI-metadata.xml" ] && echo "$0: File AMI-Metadata.xml not found under $adir/annotations." && exit 1;

#download waves

cat local/split_train.orig local/split_eval.orig local/split_dev.orig > $wdir/ami_meet_ids.flist

wgetfile=$wdir/wget_$mic.sh
manifest="wget -O $adir/MANIFEST.TXT http://groups.inf.ed.ac.uk/ami/download/temp/amiBuild-04237-Sun-Jun-15-2014.manifest.txt"
license="wget -O $adir/LICENCE.TXT http://groups.inf.ed.ac.uk/ami/download/temp/Creative-Commons-Attribution-NonCommercial-ShareAlike-2.5.txt"

echo "#!/bin/bash" > $wgetfile
echo $manifest >> $wgetfile
echo $license >> $wgetfile
while read line; do
   if [ "$mic" == "ihm" ]; then
     for m in 0 1 2 3; do
       echo "wget -c -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Headset-$m.wav" >> $wgetfile
     done
   else
     for m in $mics; do
       echo "wget -c -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Array1-0$m.wav" >> $wgetfile
     done
   fi
done < $wdir/ami_meet_ids.flist

chmod +x $wgetfile
echo "Downloading audio files for $mic scenario."
echo "Look at $wdir/log/download_ami_$mic.log for download progress"

$wgetfile &> $wdir/log/download_ami_$mic.log

echo "Downloads of AMI corpus completed succesfully. License can be found under $adir/LICENSE.TXT"

