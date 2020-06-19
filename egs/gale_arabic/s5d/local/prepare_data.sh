#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

# GALE Arabic phase 2 Conversation Speech
dir1=/export/corpora/LDC/LDC2013S02/         
dir2=/export/corpora/LDC/LDC2013S07/         
text1=/export/corpora/LDC/LDC2013T04/
text2=/export/corpora/LDC/LDC2013T17/
# GALE Arabic phase 2 News Speech
dir3=/export/corpora/LDC/LDC2014S07/         
dir4=/export/corpora/LDC/LDC2015S01/         
text3=/export/corpora/LDC/LDC2014T17/        
text4=/export/corpora/LDC/LDC2015T01/        
# GALE Arabic phase 3 Conversation Speech
dir5=/export/corpora/LDC/LDC2015S11/         
dir6=/export/corpora/LDC/LDC2016S01/         
text5=/export/corpora/LDC/LDC2015T16/        
text6=/export/corpora/LDC/LDC2016T06/        
# GALE Arabic phase 3 News Speech
dir7=/export/corpora/LDC/LDC2016S07/          
dir8=/export/corpora/LDC/LDC2017S02/          
text7=/export/corpora/LDC/LDC2016T17/         
text8=/export/corpora/LDC/LDC2017T04/         
# GALE Arabic phase 4 Conversation Speech
dir9=/export/corpora/LDC/LDC2017S15/          
text9=/export/corpora/LDC/LDC2017T12/         
# GALE Arabic phase 4 News Speech
dir10=/export/corpora/LDC/LDC2018S05/         
text10=/export/corpora/LDC/LDC2018T14/        

mgb2_dir=""
process_xml=""
mer=80

. ./utils/parse_options.sh

gale_data=GALE

mkdir -p $gale_data 
# check that sox is installed 
which sox  &>/dev/null
if [[ $? != 0 ]]; then 
 echo "$0: sox is not installed"; exit 1
fi

for dvd in $dir1 $dir2 $dir3 $dir4 $dir5 $dir6 $dir7 $dir8 $dir9 $dir10; do
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

for cdx in $text1 $text2 $text3 $text4 $text5 $text6 $text7 $text8 $text9 $text10; do
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

# prepare MGB2 data
if [ ! -z $mgb2_dir ]; then
  echo "preparing MGB2 data"

  xmldir=$mgb2_dir/train/xml/bw
  output_dir=$gale_data/mgb2
  mkdir -p $output_dir

  if [ -f $output_dir/wav.scp ]; then
    mkdir -p $output_dir/.backup
    mv $output_dir/wav.scp ${output_dir}/.backup
    mv $output_dir/mgb2 ${output_dir}/.backup
  fi

  if [ $process_xml == 'python' ]; then
    echo "using python to process xml file"
    # check if bs4 and lxml are installed in python
    local/check_tools.sh
    ls $mgb2_dir/train/wav/ | while read name; do
      basename=`basename -s .wav $name`
      [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
      local/process_xml.py $xmldir/$basename.xml - | local/add_to_datadir.py $basename $train_dir $mer
      echo $basename $db_dir/train/wav/$basename.wav >> $output_dir/wav.scp
    done
  elif [ $process_xml == 'xml' ]; then
    # check if xml binary exsits
    if command -v xml >/dev/null 2>/dev/null; then
      echo "using xml"
      ls $mgb2_dir/train/wav/ | while read name; do
        basename=`basename -s .wav $name`
        [ ! -e $xmldir/$basename.xml ] && echo "Missing $xmldir/$basename.xml" && exit 1
        xml sel -t -m '//segments[@annotation_id="transcript_align"]' -m "segment" -n -v  "concat(@who,' ',@starttime,' ',@endtime,' ',@WMER,' ')" -m "element" -v "concat(text(),' ')" $xmldir/$basename.xml | local/add_to_datadir.py $basename $output_dir $mer
        echo $basename $db_dir/train/wav/$basename.wav >> $output_dir/wav.scp
      done
    else
      echo "xml not found, you may use python by '--process-xml python'"
      exit 1;
    fi
  else
    # invalid option
    echo "$0: invalid option for --process-xml, choose from 'xml' or 'python'"
    exit 1;
  fi

  # add mgb2 data to training data (GALE/all and wav.scp)
  mv $gale_data/all $gale_data/all.gale 
  cat $gale_data/all.gale $output_dir/mgb2 > $gale_data/all
  cat $output_dir/wav.scp >> $gale_data/wav.scp

  # for dict preparation 
  grep -v -f local/test/dev_all $gale_data/all.gale | \
         grep -v -f local/test/test_p2 | \
         grep -v -f local/test/mt_eval_all | \
         grep -v -f local/bad_segments > $gale_data/all.gale.train 
  awk '{printf $2 " "; for (i=5; i<=NF; i++) {printf $i " "} printf "\n"}' $gale_data/all.gale.train | sort -u > $gale_data/gale_text
echo "$0:MGB2 data added to training data"
fi


echo "$0:data prep text succeeded"

mkdir -p data
dir=$(utils/make_absolute.sh data/)
grep -f local/test/dev_all $gale_data/all | grep -v -f local/bad_segments > $gale_data/all.dev
grep -f local/test/test_p2 $gale_data/all | grep -v -f local/bad_segments > $gale_data/all.test_p2
grep -f local/test/mt_eval_all $gale_data/all | grep -v -f local/bad_segments > $gale_data/all.mt_all
grep -v -f local/test/dev_all $gale_data/all | \
       grep -v -f local/test/test_p2 | \
       grep -v -f local/test/mt_eval_all | \
       grep -v -f local/bad_segments > $gale_data/all.train 

for x in dev test_p2 mt_all train; do
 outdir=data/$x
 file=$gale_data/all.$x 
 mkdir -p $outdir
 awk '{print $2 " " $2}' $file | sort -u > $outdir/utt2spk 
 cp -pr $outdir/utt2spk $outdir/spk2utt
 awk '{print $2 " " $1 " " $3 " " $4}' $file  | sort -u > $outdir/segments
 awk '{printf $2 " "; for (i=5; i<=NF; i++) {printf $i " "} printf "\n"}' $file | sort -u > $outdir/text
done 

grep -f local/test/dev_all $gale_data/wav.scp > $dir/dev/wav.scp
grep -f local/test/test_p2 $gale_data/wav.scp > $dir/test_p2/wav.scp
grep -f local/test/mt_eval_all $gale_data/wav.scp > $dir/mt_all/wav.scp

cat $gale_data/wav.scp | awk -v seg=$dir/train/segments 'BEGIN{while((getline<seg) >0) {seen[$2]=1;}}
 {if (seen[$1]) { print $0}}' > $dir/train/wav.scp
 
echo "$0:data prep split succeeded"
exit 0
