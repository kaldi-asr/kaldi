#!/bin/bash

lp=
lr=
ar=
split=BaEval
version=1
relname=
exp=c
cer=0
dryrun=true
dir="exp/sgmm5_mmi_b0.1/"
final=false
dev2shadow=dev10h.uem
eval2shadow=eval.uem
team=RADICAL

#end of configuration

echo $0 " " "$@"

[ -f ./cmd.sh ] && . ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 2 ] ; then
  echo "Invalid number of parameters!"
  echo "Parameters " "$@"
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
  set -x
  source_file=$1
  target_file=$2
  if [ ! -f $source_file ] ; then
    echo "The file $source_file does not exist!"
    exit 1
  else
    if [ ! -f $target_file ] ; then
      if ! $dryrun ; then
        ln -s `readlink -f $source_file` $target_file || exit 1
      fi
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
    fdate=`stat --printf='%y' $source_xml`
    echo "The source file $source_xml has timestamp of $fdate"
    echo "Authorizing empty terms from `basename $kwlist`..."
    if ! $dryrun ; then
      local/fix_kwslist.pl $kwlist $source_xml $fixed_xml || exit 1
    else
      fixed_xml=$source_xml
    fi
    echo "Exporting...export_file $fixed_xml $export_xml "
    export_file $fixed_xml $export_xml || exit 1
  else
    echo "The file $source_xml does not exist. Exiting..."
    exit 1
  fi
  echo "Export done successfully..."
  return 0
}

if [[ "$eval_kwlist_file" == *.kwlist.xml ]] ; then
  corpus=`basename $eval_kwlist_file .kwlist.xml`
elif [[ "$eval_kwlist_file" == *.kwlist2.xml ]] ; then
  corpus=`basename $eval_kwlist_file .kwlist2.xml`
else
  echo "Unknown naming pattern of the kwlist file $eval_kwlist_file"
  exit 1
fi
#REMOVE the IARPA- prefix, if present
#corpus=${corpora##IARPA-}

scores=`find -L $dir  -name "sum.txt"  -path "*${dev2shadow}_${eval2shadow}*" | xargs grep "|   Occurrence" | cut -f 1,13 -d '|'| sed 's/:|//g' | column -t | sort -k 2 -n -r  `
[ -z "$scores" ] && echo "Nothing to export, exiting..." && exit 1

echo  "$scores" | head
count=`echo "$scores" | wc -l`
echo "Total result files: $count"
best_score=`echo "$scores" | head -n 1 | cut -f 1 -d ' '`

lmwt=`echo $best_score | sed 's:.*/kws_\([0-9][0-9]*\)/.*:\1:g'`
echo "Best scoring file: $best_score"
echo $lmwt
base_dir=`echo $best_score | sed "s:\\(.*\\)/${dev2shadow}_${eval2shadow}/.*:\\1:g"`
echo $base_dir

eval_dir=$base_dir/$eval2shadow/kws_$lmwt/
eval_kwlist=$eval_dir/kwslist.xml
eval_fixed_kwlist=$eval_dir/kwslist.fixed.xml
eval_export_kwlist=$outputdir/KWS13_${team}_${corpus}_${split}_KWS_${lp}_${lr}_${ar}_${relname}_${version}.kwslist.xml

echo "export_kws_file $eval_kwlist $eval_fixed_kwlist $eval_kwlist_file $eval_export_kwlist"
export_kws_file $eval_kwlist $eval_fixed_kwlist $eval_kwlist_file $eval_export_kwlist

echo "Everything looks fine, good luck!"
exit 0

