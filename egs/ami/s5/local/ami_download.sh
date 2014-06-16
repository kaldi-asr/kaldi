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

mkdir -p $adir/amicorpus

#download annotations
annot="$adir/ami_public_manual_1.6.zip"
if [[ ! -d $adir/annotations || ! -f "$annot" ]]; then
  echo "Downloading annotiations..."
  wget -O $annot $amiurl/AMICorpusAnnotations/ami_public_manual_1.6.zip
  mkdir $adir/annotations
  unzip -d $adir/annotations $annot &> /dev/null
fi
[ ! -f "$adir/annotations/AMI-metadata.xml" ] && echo "$0: File AMI-Metadata.xml not found under $adir/annotations." && exit 1;

#download waves
ihm_template="wget -P amicorpus/IB4011/audio http://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/IB4011/audio/IB4011.Headset-3.wav"
license="wget http://groups.inf.ed.ac.uk/ami/download/temp/amiBuild-04237-Sun-Jun-15-2014.manifest.txt
wget http://groups.inf.ed.ac.uk/ami/download/temp/Creative-Commons-Attribution-NonCommercial-ShareAlike-2.5.txt"

wgetfile=$adir/wget_$mic.sh

echo "#!/bin/bash" > $wgetfile
echo $license >> $wgetfile

cat local/split_train.orig local/split_eval.orig local/split_dev.orig > $adir/ami_file_ids.flist

if [ "$mic" == "ihm" ]; then
  while read line; do
     for hid in 0 1 2 3; do
       echo "wget -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Headset-$hid.wav" >> $wgetfile
     done
  done < $adir/ami_file_ids.flist
elif [ "$mic" == "sdm" ]; then

elif [ "$mic" == "mdm" ]; then

else
  exit 1;
fi

#chmod +x $wgetfile
#. $wgetfile &> $adir/log/download$mic.log

