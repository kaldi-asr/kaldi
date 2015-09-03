#!/bin/bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

echo $0 "$@"

galeData=$(readlink -f "${@: -1}" ); 

length=$(($#-1))
args=${@:1:$length}

top_pwd=`pwd`
txtdir=$galeData/txt 
mkdir -p $txtdir; cd $txtdir

for cdx in ${args[@]}; do
  echo "Preparing $cdx"
  if [[ $cdx  == *.tgz ]] ; then
     tar -xvf $cdx
  elif [  -d "$cdx" ]; then
    ln -s $cdx `basename $cdx`
  else
    echo "I don't really know what I shall do with $cdx " >&2 
  fi
done

find -L . -type f -name *.tdf | while read file; do 
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


perl ${top_pwd}/local/normalize_transcript_BW.pl contentall.tmp$$ contentall.buck.tmp$$

paste allid.tmp$$ contentall.buck.tmp$$ | sed 's: $::' | awk '{if (NF>5) {print $0}}'  > all_1.tmp$$

awk '{$1="";print $0}' all_1.tmp$$ | sed 's:^ ::' > $galeData/all
awk '{if ($1 == "report") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' >  $galeData/report
awk '{if ($1 == "conversational") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' > $galeData/conversational

#cd ..;
#rm -fr $txtdir
cd $top_pwd
echo data prep text succeeded
