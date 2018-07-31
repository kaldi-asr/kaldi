#!/bin/bash

# Copyright  2014  University of Edinburgh (Author: Pawel Swietojanski, Jonathan Kilgour)
#            2015  Brno University of Technology (Author: Karel Vesely)
#            2016  Johns Hopkins University (Author: Daniel Povey)
#


. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <mic> <ami-dir>"
  echo " where <mic> is either ihm, sdm1 or mdm8, and <ami-dir> is download space."
  echo "e.g.: $0 sdm1 /foo/bar/AMI"
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  exit 1;
fi
mic=$1
adir=$2

amiurl=http://groups.inf.ed.ac.uk/ami
#annotver=ami_public_manual_1.6.1
wdir=data/local/downloads

case $mic in
  ihm)
  ;;
  mdm8) mics="1 2 3 4 5 6 7 8"
  ;;
  sdm1) mics="1"
  ;;
  *) echo "Wrong 'mic' option $mic" && exit 1
  ;;
esac
echo "mics set to '$mics'"

mkdir -p $adir
mkdir -p $wdir/log

#download annotations
#
#annot="$adir/$annotver"
#if [[ ! -d $adir/annotations || ! -f "$annot" ]]; then
#  echo "Downloading annotiations..."
#  wget -nv -O $annot.zip $amiurl/AMICorpusAnnotations/$annotver.zip &> $wdir/log/download_ami_annot.log
#  mkdir -p $adir/annotations
#  unzip -o -d $adir/annotations $annot.zip &> /dev/null
#fi
#[ ! -f "$adir/annotations/AMI-metadata.xml" ] && echo "$0: File AMI-Metadata.xml not found under $adir/annotations." && exit 1;

#download waves

cat local/split_train.orig local/split_eval.orig local/split_dev.orig > $wdir/ami_meet_ids.flist

wgetfile=$wdir/wget_$mic.sh

# TODO fix this with Pawel, files don't exist anymore,
manifest="wget --continue -O $adir/MANIFEST.TXT http://groups.inf.ed.ac.uk/ami/download/temp/amiBuild-04237-Sun-Jun-15-2014.manifest.txt"
license="wget --continue -O $adir/LICENCE.TXT http://groups.inf.ed.ac.uk/ami/download/temp/Creative-Commons-Attribution-NonCommercial-ShareAlike-2.5.txt"

echo "#!/bin/bash" > $wgetfile
echo $manifest >> $wgetfile
echo $license >> $wgetfile
while read line; do
   if [ "$mic" == "ihm" ]; then
     extra_headset= #some meetings have 5 sepakers (headsets)
     for mtg in EN2001a EN2001d EN2001e; do
       [ "$mtg" == "$line" ] && extra_headset=4;
     done
     for m in 0 1 2 3 $extra_headset; do
       # Hint: avoiding re-download by '--continue',
       echo "wget -nv --continue -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Headset-$m.wav" >> $wgetfile
     done
   else
     for m in $mics; do
       # Hint: avoiding re-download by '--continue',
       echo "wget -nv --continue -P $adir/$line/audio $amiurl/AMICorpusMirror/amicorpus/$line/audio/$line.Array1-0$m.wav" >> $wgetfile
     done
   fi
done < $wdir/ami_meet_ids.flist

chmod +x $wgetfile
echo "Downloading audio files for $mic scenario."
echo "Look at $wdir/log/download_ami_$mic.log for progress"
$wgetfile &> $wdir/log/download_ami_$mic.log

# Do rough check if #wavs is as expected, it will fail anyway in data prep stage if it isn't,
if [ "$mic" == "ihm" ]; then
  num_files=$(find $adir -iname *Headset* | wc -l)
  if [ $num_files -ne 687 ]; then
    echo "Warning: Found $num_files headset wavs but expected 687. Check $wdir/log/download_ami_$mic.log for details."
    exit 1;
  fi
else
  num_files=$(find $adir -iname *Array1* | wc -l)
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



