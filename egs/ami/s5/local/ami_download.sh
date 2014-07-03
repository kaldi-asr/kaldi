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

extra_headset= #EN2001{a,d,e} have 5 sepakers
echo "#!/bin/bash" > $wgetfile
echo $manifest >> $wgetfile
echo $license >> $wgetfile
while read line; do
   if [ "$mic" == "ihm" ]; then
     extra_headset= #some meetings have 5 sepakers
     for mtg in EN2001a EN2001d EN2001e; do
       [ "$mtg" == "$line" ] && extra_headset=4;
     done
     for m in 0 1 2 3 $extra_headset; do
       echo "wget -nv -c -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Headset-$m.wav" >> $wgetfile
     done
   else
     for m in $mics; do
       echo "wget -nv -c -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Array1-0$m.wav" >> $wgetfile
     done
   fi
done < $wdir/ami_meet_ids.flist

chmod +x $wgetfile
echo "Downloading audio files for $mic scenario."
echo "Look at $wdir/log/download_ami_$mic.log for download progress"
$wgetfile &> $wdir/log/download_ami_$mic.log

#do rough check if #wavs is as expected, it will fail anyway in data prep stage if it isn't
if [ "$mic" == "ihm" ]; then
  num_files=`find $adir -iname *Headset*`
  if [ $num_files -ne 687 ]; then
    echo "Warning: Found $num_files headset wavs but expected 687. Check $wdir/log/download_ami_$mic.log for details."
    exit 1;
  fi
else
  num_files=`find $adir -iname *Array1*`
  if [[ $num_files -lt 1352 && "$mic" == "mdm" ]]; then
    echo "Warning: Found $num_files distant Array1 waves but expected 1352 for mdm. Check $wdir/log/download_ami_$mic.log for details."
    exit 1;
  elif [[ $num_files -lt 169 && "$mic" == "sdm" ]]; then
    echo "Warning: Found $num_files distant Array1 waves but expected 169 for sdm. Check $wdir/log/download_ami_$mic.log for details."
    exit 1;
  fi
fi

echo "Downloads of AMI corpus completed succesfully. License can be found under $adir/LICENCE.TXT"
exit 0;



