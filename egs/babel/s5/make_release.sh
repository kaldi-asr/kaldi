#!/bin/bash

lp=
lr=
ar=
version=1
relname=
cer=0
#end of configuration

[ -f ./cmd.sh ] && . ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
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

[ ! -f $1 ] && echo "Configuration $1 does not exist! " && exit 1
. $1
outputdir=$2

function export_file {
  source_file=$1
  target_file=$2
  if [ ! -f $source_file ] ; then
    echo "The file $source_file does not exist!"
    exit 1
  else
    if [ ! -f $target_file ] ; then
      ln -s `readlink -f $source_file` $target_file || exit 1
    else
      echo "The file is already there, not doing anything. Either change the version (using --version), or delete that file manually)"
      exit 1
    fi
  fi
  return 0
}

function export_kws_file {
  source_xml=$1
  fixed_xml=$2
  kwlist=$3
  export_xml=$4
  
  echo "Exporting KWS $source_xml as `basename $export_xml`"
  if [ -f $source_xml ] ; then
    cp $source_xml $fixed_xml.bak
    echo "Authorizing empty terms from `basename $kwlist`..."
    local/fix_kwslist.pl $kwlist $source_xml $fixed_xml || exit 1
    echo "Exporting..."
    export_file $fixed_xml $export_xml || exit 1
  else
    echo "The file $source_xml does not exist. Exiting..."
    exit 1
  fi
  echo "Export done successfully..."
  return 0
}

corpora=`basename $test_data_kwlist .kwlist.xml`


scores=`find exp/sgmm5_mmi_b0.1/  -name "sum.txt"  -path "*eval.uem*"      | xargs grep "|   Occurrence" | cut -f 1,13 -d '|'| sed 's/:|//g' | column -t | sort -k 2 -n -r  | head`
[ -z "$scores" ] && echo "Nothing to export, exiting..." && exit 1

echo  "$scores"
ii=`echo "$scores" | head -n 1 | cut -f 1 -d ' '`


dev_kwlist=`echo $ii | sed "s/sum.txt/kwslist.xml/"`
odev_kwlist=${dev_kwlist%.xml}.fixed.xml
filename="KWS13_RADICAL_${corpora}_BaDev_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.kwlist.xml"
export_kws_file $dev_kwlist $odev_kwlist $test_data_kwlist $outputdir/$filename || exit 1

eval_kwlist=`echo $dev_kwlist | sed "s:eval.uem:test.uem:g"`
oeval_kwlist=${eval_kwlist%.xml}.fixed.xml
filename="KWS13_RADICAL_${corpora}_BaEval_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.kwlist.xml"
export_kws_file $eval_kwlist $oeval_kwlist $test_data_kwlist $outputdir/$filename || exit 1

dev_kwlist=${dev_kwlist%.xml}.unnormalized.xml
odev_kwlist=${dev_kwlist%.xml}.unnormalized.fixed.xml
filename="KWS13_RADICAL_${corpora}_BaDev_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.unnormalized.kwlist.xml"
export_kws_file $dev_kwlist $odev_kwlist $test_data_kwlist $outputdir/$filename || exit 1

eval_kwlist=${eval_kwlist%.xml}.unnormalized.xml
oeval_kwlist=${eval_kwlist%.xml}.unnormalized.fixed.xml
filename="KWS13_RADICAL_${corpora}_BaEval_KWS_${lp}_${lr}_${ar}_c-${relname}_${version}.unnormalized.kwlist.xml"
export_kws_file $eval_kwlist $oeval_kwlist $test_data_kwlist $outputdir/$filename || exit 1


if [ $cer -eq 1 ] ; then
  scores=`find exp/sgmm5_mmi_b0.1 -name "*char.ctm.sys" -ipath "*eval.uem*" | xargs grep 'Sum/Avg' | sed 's/:* *| */ /g' | sed 's/  */ /g' | sort  -n -k 9 | column -t | head`
else
  scores=`find exp/sgmm5_mmi_b0.1 -name "*.ctm.sys" -not -name "*char.ctm.sys" -ipath "*eval.uem*" | xargs grep 'Sum/Avg' | sed 's/:* *| */ /g' | sed 's/  */ /g' | sort  -n -k 9 | column -t | head`
fi
[ -z $scores ] && echo "Nothing to export, exiting..." && exit 1

echo  "$scores"
ii=`echo "$scores" | head -n 1 | cut -f 1 -d ' '`

dev_sttlist=`echo $ii | sed "s/char.ctm/ctm/" | sed "s/ctm.sys/ctm/"`
filename="KWS13_RADICAL_${corpora}_BaDev_STT_${lp}_${lr}_${ar}_c-${relname}_${version}.ctm"
echo  "Exporting STT BaDev $dev_sttlist as $filename "
export_file $dev_sttlist $outputdir/$filename || exit 1

eval_sttlist=`echo $dev_sttlist | sed "s:eval.uem:test.uem:g"`
filename="KWS13_RADICAL_${corpora}_BaEval_STT_${lp}_${lr}_${ar}_c-${relname}_${version}.ctm"
echo  "Exporting STT BaEval $eval_sttlist as $filename "
export_file $eval_sttlist $outputdir/$filename || exit 1

echo "Everything looks fine, good luck!"
exit 0

