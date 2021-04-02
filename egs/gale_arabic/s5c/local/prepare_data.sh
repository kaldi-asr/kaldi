#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

dir1=/export/corpora/LDC/LDC2013S02/
dir2=/export/corpora/LDC/LDC2013S07/
dir3=/export/corpora/LDC/LDC2014S07/
text1=/export/corpora/LDC/LDC2013T17/
text2=/export/corpora/LDC/LDC2013T04/
text3=/export/corpora/LDC/LDC2014T17/
gale_data=GALE

mkdir -p $gale_data 
# check that sox is installed 
which sox  &>/dev/null
if [[ $? != 0 ]]; then 
 echo "$0: sox is not installed"; exit 1
fi

for dvd in $dir1 $dir2 $dir3; do
  dvd_full_path=$(utils/make_absolute.sh $dvd)
  if [[ ! -e $dvd_full_path ]]; then 
    echo "$0: missing $dvd_full_path"; exit 1;
  fi
  find $dvd_full_path \( -name "*.wav" -o -name "*.flac" \)  | while read file; do
    id=$(basename $file | awk '{gsub(".wav","");gsub(".flac","");print}')
    echo "$id sox $file -r 16000 -t wav - |"
  done 
done | sort -u > $gale_data/wav.scp
echo "$0:data prep audio succeded"

gale_data=$(utils/make_absolute.sh "GALE" );
top_pwd=`pwd`
txtdir=$gale_data/txt
mkdir -p $txtdir; cd $txtdir

for cdx in $text1 $text2 $text3; do
  echo "$0:Preparing $cdx"
  if [[ $cdx  == *.tgz ]] ; then
     tar -xvf $cdx
  elif [  -d "$cdx" ]; then
    ln -s $cdx `basename $cdx`
  else
    echo "$0:I don't really know what I shall do with $cdx " >&2
  fi
done

find -L . -type f -name "*.tdf" | while read file; do
sed '1,3d' $file  # delete the first 3 lines
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


awk '{$1="";print $0}' all_1.tmp$$ | sed 's:^ ::' > $gale_data/all
awk '{if ($1 == "report") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' >  $gale_data/report
awk '{if ($1 == "conversational") {$1="";print $0}}' all_1.tmp$$ | sed 's:^ ::' > $gale_data/conversational

cd ..;
rm -fr $txtdir
cd $top_pwd
echo "$0:dat a prep text succeeded"

mkdir -p data
dir=$(utils/make_absolute.sh data/)
grep -f local/test_list $gale_data/all | grep -v -f local/bad_segments > $gale_data/all.test
grep -v -f local/test_list $gale_data/all | grep -v -f local/bad_segments > $gale_data/all.train 

for x in test train; do
 outdir=data/$x
 file=$gale_data/all.$x 
 mkdir -p $outdir
 awk '{print $2 " " $2}' $file | sort -u > $outdir/utt2spk 
 cp -pr $outdir/utt2spk $outdir/spk2utt
 awk '{print $2 " " $1 " " $3 " " $4}' $file  | sort -u > $outdir/segments
 awk '{printf $2 " "; for (i=5; i<=NF; i++) {printf $i " "} printf "\n"}' $file | sort -u > $outdir/text
done 

grep -f local/test_list $gale_data/wav.scp > $dir/test/wav.scp

cat $gale_data/wav.scp | awk -v seg=$dir/train/segments 'BEGIN{while((getline<seg) >0) {seen[$2]=1;}}
 {if (seen[$1]) { print $0}}' > $dir/train/wav.scp
 
echo "$0:data prep split succeeded"
exit 0
