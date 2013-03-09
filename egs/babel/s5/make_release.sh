#!/bin/bash

lp=
lr=
ar=
version=1
relname=
cer=0
#end of configuration

. ./utils/parse_options.sh

if [ $# -ne 2 ] ; then
  echo "Invalid number of parameters!"
  echo "$0 --ar <NTAR|TAR> --lr <BaseLR|BabelLR|OtherLR> --lp <FullLP|LimitedLP> --relname <NAME> [--version <version-nr> ] <config> <output>"
  exit 1
fi


[ -z $lp ] && echo "Error -- you must specify --lp <FullLP|LimitedLP>" && exit 1
if [ "$lp" != "FullLP" ] && [ "$lp" != "LimitedLP" ] ; then
  echo "Error -- you must specify --lp <FullLP|LimitedLP>" && exit 1
fi

[ -z $lr ] && echo "Error -- you must specify --lr <BaseLR|BabelLR|OtherLR>" && exit 1
if [ "$lr" != "BaseLR" ] && [ "$lr" != "BabelLR" ]  && [ "$lr" != "OtherLR" ] ; then
  echo "Error -- you must specify --lr <BaseLR|BabelLR|OtherLR>" && exit 1
fi
[ -z $ar ] && echo "Error -- you must specify --ar <NTAR|TAR>" && exit 1
if [ "$ar" != "NTAR" ] && [ "$ar" != "TAR" ] ; then
  echo "Error -- you must specify --ar <NTAR|TAR>" && exit 1
fi
[ -z $relname ] && echo "Error -- you must specify name" && exit 1

. $1
outputdir=$2
corpora=`basename $test_data_kwlist .kwlist.xml`


scores=`find exp/sgmm5_mmi_b0.1/  -name "sum.txt"  -path "*eval.uem*"      | xargs grep "|   Occurrence" | cut -f 1,13 -d '|'| sed 's/:|//g' | column -t | sort -k 2 -n -r  | head`

echo  "$scores"

ii=`echo "$scores" | head -n 1 | cut -f 1 -d ' '`


dev_kwlist=`echo $ii | sed "s/sum.txt/kwslist.xml/"`
eval_kwlist=`echo $dev_kwlist | sed "s:eval.uem:test.uem:g"`
odev_kwlist=${dev_kwlist%.xml}.fixed.xml
oeval_kwlist=${eval_kwlist%.xml}.fixed.xml


echo $dev_kwlist 
echo $eval_kwlist

[ -f $dev_kwlist ] && cp $dev_kwlist $dev_kwlist.bak
[ -f $eval_kwlist ] && cp $eval_kwlist $eval_kwlist.bak

local/fix_kwslist.pl $test_data_kwlist $dev_kwlist  ${odev_kwlist}
local/fix_kwslist.pl $test_data_kwlist $eval_kwlist  ${oeval_kwlist}

filename="KWS13_RADICAL_${corpora}_BaDev_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.kwlist.xml"
echo $filename
ln -s `readlink -f $odev_kwlist` $outputdir/$filename
filename="KWS13_RADICAL_${corpora}_BaEval_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.kwlist.xml"
echo $filename
ln -s `readlink -f $oeval_kwlist` $outputdir/$filename

dev_kwlist=${dev_kwlist%.xml}.unnormalized.xml
eval_kwlist=${eval_kwlist%.xml}.unnormalized.xml
odev_kwlist=${dev_kwlist%.xml}.unnormalized.fixed.xml
oeval_kwlist=${eval_kwlist%.xml}.unnormalized.fixed.xml
echo $dev_kwlist 
echo $eval_kwlist

[ -f $dev_kwlist ] && cp $dev_kwlist $dev_kwlist.bak
[ -f $eval_kwlist ] && cp $eval_kwlist $eval_kwlist.bak
local/fix_kwslist.pl $test_data_kwlist $dev_kwlist  ${odev_kwlist}
local/fix_kwslist.pl $test_data_kwlist $eval_kwlist  ${oeval_kwlist}

filename="KWS13_RADICAL_${corpora}_BaDev_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.unnormalized.kwlist.xml"
echo $filename
ln -s `readlink -f $odev_kwlist` $outputdir/$filename
filename="KWS13_RADICAL_${corpora}_BaEval_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.unnormalized.kwlist.xml"
echo $filename
ln -s `readlink -f $oeval_kwlist` $outputdir/$filename


if [ $cer -eq 1 ] ; then
  scores=`find exp/sgmm5_mmi_b0.1 -name "*char.ctm.sys" -ipath "*eval.uem*" | xargs grep 'Sum/Avg' | sed 's/:* *| */ /g' | sed 's/  */ /g' | sort  -n -k 9 | column -t | head`
else
  scores=`find exp/sgmm5_mmi_b0.1 -name "*.ctm.sys" -not -name "*char.ctm.sys" -ipath "*eval.uem*" | xargs grep 'Sum/Avg' | sed 's/:* *| */ /g' | sed 's/  */ /g' | sort  -n -k 9 | column -t | head`
fi

echo  "$scores"

ii=`echo "$scores" | head -n 1 | cut -f 1 -d ' '`
dev_sttlist=`echo $ii | sed "s/char.ctm/ctm/" | sed "s/ctm.sys/ctm/"`
eval_sttlist=`echo $dev_sttlist | sed "s:eval.uem:test.uem:g"`

echo $dev_sttlist 
echo $eval_sttlist

[ -f $dev_sttlist.bak ] && cp $dev_sttlist $dev_sttlist.bak
[ -f $eval_sttlist.bak ] && cp $eval_sttlist $eval_sttlist.bak

filename="KWS13_RADICAL_${corpora}_BaDev_STT_${lp}_${lr}_${ar}_c-${relname}_${version}.ctm"
echo $filename
ln -s `readlink -f $dev_sttlist` $outputdir/$filename
filename="KWS13_RADICAL_${corpora}_BaEval_STT_${lp}_${lr}_${ar}_c-${relname}_${version}.ctm"
echo $filename
ln -s `readlink -f $eval_sttlist` $outputdir/$filename



