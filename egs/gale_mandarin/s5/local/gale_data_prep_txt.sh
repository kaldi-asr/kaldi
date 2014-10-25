#!/bin/bash

# Copyright 2014 (author: Ahmed Ali, Hainan Xu)
# Apache 2.0

if [ $# -ne 2 ]; then
   echo "Arguments should be the <gale folder> <txt data folder>"; exit 1
fi

export LC_ALL=C

galeData=$1
text=$2

cur=`pwd`

txtdir=$galeData/txt 
mkdir -p $galeData/txt 

cd $txtdir

find $text -type f -name *.tdf | while read file; do 
sed '1,3d' $file 
done > all.tmp


perl -e '
    ($inFile,$idFile,$txtFile,$spk,$mapf)= split /\s+/, $ARGV[0];
    open(IN, "$inFile");
    open(ID, ">$idFile");
    open(TXT, ">$txtFile");
    open(SPK, ">$spk");
    open(MAP, ">$mapf");
    while (<IN>) {
      @arr= split /\t/,$_;
      $arr[4] =~ s/ //g;
      $spkid = "$arr[0]_$arr[4]";
      $spkfix = sprintf("%060s", $spkid);
      $start=sprintf ("%0.3f",$arr[2]);$rStart=$start;$start=~s/\.//; $start=~s/^0+$/0/; $start=~s/^0+([^0])/$1/; # remove zeros at the beginning
      $end=sprintf ("%0.3f",$arr[3]);$rEnd=$end;$end=~s/^0+([^0])/$1/;$end=~s/\.//;
	  $id="$arr[11] $arr[0] ${spkfix}_$arr[0]_${start}_${end} $rStart $rEnd\n";
	  next if ($rStart == $rEnd);
	  $id =~ s/.sph//g;
	  print ID $id;
      print TXT "$arr[7]\n";
      print SPK "${spkfix}_$arr[0]_${start}_${end} ${spkfix}\n";
      print MAP "$arr[0] ${spkfix}_$arr[0]\n";
 }' "all.tmp allid.tmp contentall.tmp utt2spk.tmp map.tmp" 

perl -p -i -e 's=/.$==g' contentall.tmp

cd $cur


pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz 
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
  python setup.py build 
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi

cat $txtdir/contentall.tmp |\
  sed -e 's/,//g' |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  perl -pe 's/<Event.*?>/ /g' |\
  sed -e 's/\[NS\]//g' |\
  sed -e 's/\[ns\]//g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/gale_normalize.pl | \
  python local/gale_segment.py \
  > $txtdir/text

paste $txtdir/allid.tmp $txtdir/text | sed 's: $::' | awk '{if (NF>5) {print $0}}'  > $txtdir/all_1.tmp

awk '{$1="";print $0}' $txtdir/all_1.tmp | sed 's:^ ::' > $txtdir/../all

cat $txtdir/utt2spk.tmp | sort -u > $txtdir/../utt2spk
cat $txtdir/map.tmp | sort -u > $txtdir/../map

sort -c $txtdir/../utt2spk 

utils/utt2spk_to_spk2utt.pl $txtdir/../utt2spk | sort -u > $txtdir/../spk2utt

cd ..;
rm -fr $txtdir

echo data prep text succeeded
