#!/bin/bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

if [ $# -ne 3 ]; then
   echo "Arguments should be the <gale folder> <txt CD1> <txt CD2>"; exit 1
fi

galeData=$(readlink -f $1)
cd1=$(readlink -f $2)
cd2=$(readlink -f $3)

txtdir=$galeData/txt 
mkdir -p $galeData/txt 

cd $txtdir
for dir in $cd1 $cd2; do
 tar -xvf $dir
done

find . -type f -name *.tdf | while read file; do 
sed '1,3d' $file 
done >  all.tmp$$


perl -e '
    ($inFile,$idFile,$txtFile)= split /\s+/, $ARGV[0];
    open(IN, "$inFile");
    open(ID, ">$idFile");
    open(TXT, ">$txtFile");
    while (<IN>) {
      @arr= split /\t/,$_;
      $start=sprintf ("%0.3f",$arr[2]);$rStart=$start;$start=~s/\.//; $start=~s/^0+$/0/; $start=~s/^0+([^0])/$1/; # remove zeros at the beginning
      $end=sprintf ("%0.3f",$arr[3]);$rEnd=$end;$end=~s/^0+([^0])/$1/;$end=~s/\.//;
	  if ( ($arr[11] !~ m/report/) && ($arr[11] !~ m/conversational/) ){$arr[11]="UNK";}
	  $id="$arr[11] $arr[0] $arr[0]_${start}_${end} $rStart $rEnd\n";
	  next if ($rStart == $rEnd);
	  $id =~ s/.sph//g;
	  print ID $id;
      print TXT "$arr[7]\n";
 }' "all.tmp$$ allid.tmp$$ contentall.tmp$$" 


perl ../../local/normalize_transcript_BW.pl contentall.tmp$$ contentall.buck.tmp$$

paste allid.tmp$$ contentall.buck.tmp$$ | sed 's: $::' | awk '{if (NF>5) {print $0}}'  > all_1.tmp$$

awk '{$1="";print $0}' all_1.tmp$$ | sed 's:^ ::' > $galeData/all
awk '{if ($1 == "report") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' >  $galeData/report
awk '{if ($1 == "conversational") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' > $galeData/conversational

cd ..;
rm -fr $txtdir

echo data prep text succeeded
