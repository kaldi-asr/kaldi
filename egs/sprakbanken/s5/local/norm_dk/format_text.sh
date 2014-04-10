#!/bin/bash

#dir=norm_dk

#dos2unix $2

mode=$1

dir=/home/ask/kaldi-trunk/egs/att/s5/local/norm_dk

abbr=$dir/anot.tmp
rem=$dir/rem.tmp
line=$dir/line.tmp
num=$dir/num.tmp
nonum=$dir/nonum.tmp

$dir/expand_abbr_medical.sh $2 > $abbr;
$dir/remove_annotation.sh $abbr > $rem;
if [ $mode != "am" ]; then
    $dir/sent_split.sh $rem > $line;
else
    $dir/write_out_formatting.sh $rem > $line;
fi

$dir/expand_dates.sh $line |\
$dir/format_punct.sh  >  $num;
python3 $dir/writenumbers.py $dir/numbersUp.tbl $num $nonum;
cat $nonum | $dir/write_punct.sh | \
perl -pi -e "s/^\n//" | PERLIO=:utf8 perl -pe '$_=uc'  #> ../${2%.*}.exp.txt

#./expand_abbr_medical.sh $1 > anot.tmp
#echo "Abbreviations expanded"
#./remove_annotation.sh anot.tmp > rem.tmp
#echo "Annotations removed"
#./sent_split.sh rem.tmp > line.tmp
#echo "Sentence splitting done"
#./expand_dates.sh line.tmp |\
#echo "Uniform date formatting completed"
#./format_punct.sh  |\
#echo "Punctuation formatting completed"
#perl -pi -e "s/^\n//" > ${@%.*}.exp.txt
#echo "Output has been uppercased"

# Comment this line for debugging
wait
#rm -f $abbr $rem $line $num
