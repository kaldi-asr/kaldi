#!/usr/bin/env bash

team=RADICAL
corpusid=
partition=
scase=BaEval  #BaDev|BaEval
master=
version=1
sysid=
prim=c
cer=0
dryrun=true
dir="exp/sgmm5_mmi_b0.1/"
data=data/dev10h.seg
master=dev10h
extrasys=
final=false

#end of configuration


echo $0 " " "$@"

[ -f ./cmd.sh ] && . ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. ./utils/parse_options.sh

. $1
outputdir=$2

set -e
set -o pipefail

function submit_to_google {
  SYSPATH=$1
  #curl 'https://docs.google.com/forms/d/1MV4gf-iVOX79ZEAekEiLIo7L_UVrJnoPjdtICK5F-nc/formResponse' \
  #    --data 'entry.1721972547='$MTWV'&entry.485509816='$ATWV'&entry.694031153='$RESPATH'&entry.1851048707='$(whoami)'&submit=Submit' \
  #    --compressed
  curl -sS 'https://docs.google.com/forms/d/1MV4gf-iVOX79ZEAekEiLIo7L_UVrJnoPjdtICK5F-nc/formResponse' \
    --data 'entry.1721972547='$MTWV'&entry.485509816='$ATWV'&entry.694031153='$SYSPATH'&entry.1851048707='$(whoami)'&entry.880350279='$STWV'&entry.60995624='$OTWV'&entry.1338769660='$LatticeRecall'&entry.1333349334='$THRESHOLD'&entry.1423358838='$(pwd)'&submit=Submit' --compressed |\
  grep  --color "Your response has been recorded." || return 1
  return 0
}

function export_file {
  #set -x
  source_file=$1
  target_file=$2
  if [ ! -f $source_file ] ; then
    echo "The file $source_file does not exist!"
    exit 1
  else
    if [ ! -f $target_file ] ; then
      if ! $dryrun ; then
        ln -s `utils/make_absolute.sh $source_file` $target_file || exit 1
        ls -al $target_file
      else
        echo "$source_file -> $target_file"
      fi
      
    else
      echo "The file is already there, not doing anything. Either change the version (using --version), or delete that file manually)"
      exit 1
    fi
  fi
  #set +x
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

function find_best_kws_result {
  local dir=$1
  local mask=$2
  local record=`(find $dir -name "sum.txt" -path "$mask" -not -ipath "*rescored*" | xargs grep "^| *Occ")  | cut -f 1,13,17 -d '|' | sed 's/|//g'  | column -t | sort -r -n -k 3 | head -n 1`
  echo $record >&2
  local file=`echo $record | awk -F ":" '{print $1}'`
  #echo $file >&2
  local path=`dirname $file`
  #echo $path >&2
  echo $path
}

function find_best_stt_result {
  local dir=$1
  local mask=$2
  local record=`(find $dir -name "*.ctm.sys" -path "$mask" -not -ipath "*rescore*" | xargs grep Avg)  | sed 's/|//g' | column -t | sort -n -k 9 | head -n 1`
  
  echo $record >&2
  local file=`echo $record | awk -F ":" '{print $1}'`
  #echo $file >&2
  local path=`dirname $file`
  #echo $path >&2
  echo $path
}

function create_sysid {
  local best_one=$1
  local sysid=
  local taskid=`basename $best_one`
  local system_path=`dirname $best_one`
  if [[ $system_path =~ .*sgmm5.* ]] ; then
    sysid=PLP
  elif [[ $system_path =~ .*nnet.* ]] ; then
    sysid=DNN
  elif [[ $system_path =~ .*sgmm7.* ]] ; then
    sysid=BNF
  elif [[ $system_path =~ .*4way.* ]] ; then
    sysid=4way-comb
  else
    echo "Unknown system path ($system_path), cannot deduce the systemID" >&2
    exit 1
  fi
  if [[ $taskid == *kws_* ]] ; then
    local kwsid=${taskid//kws_*/}
    kwsid=${kwsid//_/}
    if [ -z $kwsid ]; then
      echo ${sysid}
    else
      echo ${sysid}-$kwsid
    fi
  else
    echo ${sysid}
  fi
}


function get_ecf_name {
  local best_one=$1
  local taskid=`basename $best_one`
  local kwstask=${taskid//kws_*/kws}
  local kwlist=
  #echo $kwstask
  if [ -z $kwstask ] ; then
    #echo $data/kws/kwlist.xml
    kwlist= `utils/make_absolute.sh $data/kws/kwlist.xml`
  else
    #echo $data/$kwstask/kwlist.xml
    kwlist=`utils/make_absolute.sh  $data/$kwstask/kwlist.xml`
  fi
  ecf=`head -n 1 $kwlist | grep -Po "(?<=ecf_filename=\")[^\"]*"`
  echo -e "\tFound ECF: $ecf" >&2
  echo $ecf
  return 0
}

function compose_expid {
  local task=$1
  local best_one=$2
  local extraid=$3
  echo "TASK:     $task" >&2
  echo "BEST ONE: $best_one" >&2
  echo "EXTRA ID: $extraid" >&2

  [ ! -z $extraid ] && extraid="-$extraid"
  local sysid=`create_sysid $best_one`
  echo "SYS ID: $sysid" >&2
  if [ "$task" == "KWS" ]; then
    ext="kwslist.xml"
  elif [ "$task" == "STT" ]; then
    ext="ctm"
  else
    echo "Incorrect task ID ($task) given to compose_expid function!" >&2
    exit 1
  fi
  echo "${corpusid}" >&2
  echo "${partition}" >&2
  echo "${scase}" >&2
  echo "KWS14_${team}_${corpusid}_${partition}_${scase}_${task}_${prim}-${sysid}${extraid}_$version.$ext"
  return 0
}

function figure_out_scase {
  local ecf=`basename $1`
  if [[ $ecf =~ IARPA-babel.*.ecf.xml ]] ; then
    local basnam=${ecf%%.ecf.xml}
    local scase=`echo $basnam | awk -F _ '{print $2}'`
    
    if [[ $scase =~ conv-dev(\..*)? ]]; then
      echo "BaDev"
    elif [[ $scase =~ conv-eval(\..*)? ]]; then
      echo "BaEval"
    else
      echo "WARNING: The ECF file  $ecf is probably not an official file" >&2
      echo "WARNING: Does not contain conv-dev|conv-eval ($scase)" >&2
      echo "BaDev"
      return 1
    fi
  else 
    echo "WARNING: The ECF file  $ecf is probably not an official file" >&2
    echo "WARNING: Does not match the mask IARPA-babel.*.ecf.xml" >&2
    echo "BaDev"
    return 1
  fi
  return 0
}

function figure_out_partition {
  local ecf=`basename $1`
  if [[ $ecf =~ IARPA-babel.*.ecf.xml ]] ; then
    local basnam=${ecf%%.ecf.xml}
    local scase=`echo $basnam | awk -F _ '{print $2}'`
    
    if [[ $scase =~ conv-dev(\..*)? ]]; then
      echo "conv-dev"
    elif [[ $scase =~ conv-eval(\..*)? ]]; then
      echo "conv-eval"
    else
      echo "WARNING: The ECF file  $ecf is probably not an official file" >&2
      echo "conv-dev"
      return 1
    fi
  else 
    echo "WARNING: The ECF file  $ecf is probably not an official file" >&2
    echo "conv-dev"
    return 1
  fi
  return 0
}

function figure_out_corpusid {
  local ecf=`basename $1`
  if [[ $ecf =~ IARPA-babel.*.ecf.xml ]] ; then
    local basnam=${ecf%%.ecf.xml}
    local corpusid=`echo $basnam | awk -F _ '{print $1}'`
  else
    echo "WARNING: The ECF file  $ecf is probably not an official file" >&2
    local corpusid=${ecf%%.*}
  fi
  echo $corpusid
}

mkdir -p $outputdir
extrasys_unnorm="unnorm"
if [ ! -z $extrasys ] ; then
  extrasys_unnorm="${extrasys}-unnorm"
fi

#data=data/shadow.uem
dirid=`basename $data`
kws_tasks="kws "
[ -f $data/extra_kws_tasks ] &&  kws_tasks+=`cat $data/extra_kws_tasks | awk '{print $1"_kws"}'` 
[ -d $data/compounds ] && compounds=`ls $data/compounds`

if [ -z "$compounds" ] ; then
  for kws in $kws_tasks ; do
    echo $kws
    best_one=`find_best_kws_result "$dir/decode_*${dirid}*/${kws}_*" "*"`
    sysid=`create_sysid $best_one`
    ecf=`get_ecf_name $best_one`
    scase=`figure_out_scase $ecf` || break
    partition=`figure_out_partition $ecf` || break
    corpusid=`figure_out_corpusid $ecf`

    expid=`compose_expid KWS $best_one "$extrasys"`
    echo -e "\tEXPORT NORMALIZED as: $expid"
    expid_unnormalized=`compose_expid KWS $best_one "$extrasys_unnorm"`
    echo -e "\tEXPORT UNNORMALIZED as: $expid_unnormalized"

    export_kws_file $best_one/kwslist.xml $best_one/kwslist.fixed.xml $data/$kws/kwlist.xml $outputdir/$expid
    export_kws_file $best_one/kwslist.unnormalized.xml $best_one/kwslist.unnormalized.fixed.xml $data/$kws/kwlist.xml $outputdir/$expid_unnormalized
  done
else
  [ -z $master ] && echo "You must choose the master compound (--master <compound>) for compound data set" && exit 1
  for kws in $kws_tasks ; do
    echo $kws
    best_one=`find_best_kws_result "$dir/decode_*${dirid}*/$master/${kws}_*" "*"`
    (
      eval "`cat $best_one/metrics.txt | sed  's/ *= */=/g' | sed 's/,/;/g' | sed 's/Lattice Recall/LatticeRecall/g' `"
      submit_to_google $best_one $ATWV $MTWV
    ) || echo "Submission failed!"

    
    for compound in $compounds ; do
      compound_best_one=`echo $best_one | sed "s:$master/${kws}_:$compound/${kws}_:g"`
      echo "From ($kws) $best_one going to $compound_best_one"
      echo -e "\tPREPARE EXPORT: $compound_best_one"
      sysid=`create_sysid $compound_best_one`
      #ecf=`get_ecf_name $best_one`
      ecf=`utils/make_absolute.sh $data/compounds/$compound/ecf.xml`
      scase=`figure_out_scase $ecf`
      partition=`figure_out_partition $ecf`
      corpusid=`figure_out_corpusid $ecf`
      expid=`compose_expid KWS $compound_best_one "$extrasys"`
      echo -e "\tEXPORT NORMALIZED as: $expid"
      expid_unnormalized=`compose_expid KWS $compound_best_one "$extrasys_unnorm"`
      echo -e "\tEXPORT UNNORMALIZED as: $expid_unnormalized"

      export_kws_file $compound_best_one/kwslist.xml $compound_best_one/kwslist.fixed.xml $data/$kws/kwlist.xml $outputdir/$expid
      export_kws_file $compound_best_one/kwslist.unnormalized.xml $compound_best_one/kwslist.unnormalized.fixed.xml $data/$kws/kwlist.xml $outputdir/$expid_unnormalized
    done
  done
fi

##EXporting STT -- more straightforward, because there is only one task
if [ -z "$compounds" ] ; then
  #best_one=`find_best_stt_result "$dir/decode_*${dirid}*/score_*" "*"`
  best_one=`find_best_stt_result "$dir/*${dirid}*/score_*" "*"`
  echo -e "\tERROR: I don't know how to do this, yet"
  ecf=`get_ecf_name kws`
  sysid=`create_sysid $best_one`
  scase=`figure_out_scase $ecf` || break
  partition=`figure_out_partition $ecf`
  corpusid=`figure_out_corpusid $ecf`
  expid=`compose_expid STT $best_one "$extrasys"`
  echo -e "\tEXPORT NORMALIZED as: $expid"
  export_file $best_one/${dirid}.ctm $outputdir/$expid
else
  [ -z $master ] && echo "You must choose the master compound (--master <compound>) for compound data set" && exit 1
  #best_one=`find_best_stt_result "$dir/decode_*${dirid}*/$master/score_*" "*"`
  best_one=`find_best_stt_result "$dir/*${dirid}*/$master/score_*" "*"`

  for compound in $compounds ; do
    compound_best_one=`echo $best_one | sed "s:$master/score_:$compound/score_:g"`
    echo -e "\tPREPARE EXPORT: $compound_best_one"
    sysid=`create_sysid $compound_best_one`
    #ecf=`get_ecf_name $best_one`
    ecf=`utils/make_absolute.sh $data/compounds/$compound/ecf.xml`
    scase=`figure_out_scase $ecf`
    partition=`figure_out_partition $ecf`
    corpusid=`figure_out_corpusid $ecf`
    expid=`compose_expid STT $compound_best_one $extrasys`
    echo -e "\tEXPORT NORMALIZED as: $expid"

    export_file $compound_best_one/${compound}.ctm $outputdir/$expid
  done
fi

echo "Everything looks fine, good luck!"
exit 0

