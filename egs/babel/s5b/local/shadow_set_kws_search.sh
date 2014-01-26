#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

#Fail at any unhandled non-zero error code
set -e
set -o pipefail

help_message="$0: create subset of the input directory (specified as the first directory).
                 The subset is specified by the second parameter.
                 The directory in which the subset should be created is the third parameter
             Example:
                 $0 <lang-data-dir> <decode-dir> <data-dir1> [data-dir2 [data-dir3 [ ...] ]"

# Begin configuration section.  
#acwt=0.0909091
min_lmwt=7
max_lmwt=17
duptime=0.6
cmd=run.pl
model=
skip_scoring=false
stage=0
strict=true
skip_optimization=false
max_states=150000
word_ins_penalty=0
index_only=false
ntrue_scale=0.1
# End configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ "$#" -le "2" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi


datadir=$1
langdir=$2
decodedir=$3
shift; shift; shift;
datasetA=$1
datasetB=$2


if [[ ! -d "$langdir"  ]] ; then
    echo "FATAL: the lang directory does not exist"
    exit 1;
fi
if [[ ! -d "$decodedir"  ]] ; then
    echo "FATAL: the directory with decoded files does not exist"
    exit 1;
fi

for splitdatadir in $@ ; do
    kwsdatadir=$splitdatadir/kws
    if [ ! -d "$splitdatadir"  ]  ; then
        echo "FATAL: the data directory $splitdatadir does not exist"
        exit 1;
    fi
    if [ ! -d "$kwsdatadir" ] ; then
        echo "FATAL: the data directory $kwsdatadir does not exist"
        exit 1;
    fi
    if [ ! -f "$kwsdatadir/ecf.xml"  ] ; then
        echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
        exit 1;
    fi
done

kwsdatadir=$datadir/kws

! durationA=`head -1 $datasetA/kws/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2;}'` &&
   echo "Error getting duration from $datasetA/kws/ecf.xml" && exit 1;


! durationB=`head -1 $datasetB/kws/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2;}'` &&
   echo "Error getting duration from $datasetB/kws/ecf.xml" && exit 1;

[ -z $durationA ] &&  echo "Error getting duration from $datasetA/kws/ecf.xml" && exit 1;
[ -z $durationB ] &&  echo "Error getting duration from $datasetB/kws/ecf.xml" && exit 1;

if [ ! -z "$model" ]; then
    model_flags="--model $model"
fi

mkdir -p $decodedir/kws/
if [ $stage -le 0 ] ; then
  echo "Making KWS indices..."
  if [ ! -f $decodedir/kws/.done.index ] ; then
    for lmwt in `seq $min_lmwt $max_lmwt` ; do
        kwsoutdir=$decodedir/kws_$lmwt
        mkdir -p $kwsoutdir
  
        acwt=`perl -e "print (1.0/$lmwt);"` 
        steps/make_index.sh --strict $strict --cmd "$cmd" --max-states $max_states\
          --acwt $acwt $model_flags --skip-optimization $skip_optimization \
          --word_ins_penalty $word_ins_penalty \
          $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1
    done
    touch $decodedir/kws/.done.index
  else
    echo "Assuming indexing has been aready done. If you really need to re-run "
    echo "the indexing again, delete the file $decodedir/kws/.done.index"
  fi
fi

if $index_only ; then
  echo "Indexing only was requested, existing now..."
  exit 0
fi

if [ $stage -le 1 ] ; then
  echo "Searching KWS indices..."
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      kwsoutdir=$decodedir/kws_$lmwt
      dirA=$decodedir/`basename $datasetA`/kws_$lmwt
      dirB=$decodedir/`basename $datasetB`/kws_$lmwt
      mkdir -p $dirA
      mkdir -p $dirB
      
      steps/search_index.sh --cmd "$cmd" $kwsdatadir $kwsoutdir  || exit 1

      [ ! -f $datasetA/kws/utter_id ] && echo "File $datasetA/kws/utter_id must exist!" && exit 1;
      cat $kwsoutdir/result.* | \
        grep -F -f <(cut -f 1 -d ' ' $datasetA/kws/utter_id ) |\
        grep "^KW[-a-zA-Z0-9]*-A " | \
        sed 's/^\(KW.*\)-A /\1 /g' > $dirA/results 

      [ ! -f $datasetB/kws/utter_id ] && echo "File $datasetB/kws/utter_id must exist!" && exit 1;
      cat $kwsoutdir/result.* | \
        grep -F -f <(cut -f 1 -d ' ' $datasetB/kws/utter_id ) |\
        grep "^KW[-a-zA-Z0-9]*-B " | \
        sed 's/^\(KW.*\)-B /\1 /g' > $dirB/results


      dirA=$decodedir/`basename $datasetA`_`basename $datasetB`/kws_$lmwt
      dirB=$decodedir/`basename $datasetB`_`basename $datasetA`/kws_$lmwt
      mkdir -p $dirA
      mkdir -p $dirB
      [ ! -f $datasetA/kws/utter_id ] && echo "File $datasetA/kws/utter_id must exist!" && exit 1;
      cat $kwsoutdir/result.* | \
        grep -F -f <(cut -f 1 -d ' ' $datasetA/kws/utter_id ) |\
        grep "^KW[-a-zA-Z0-9]*-B " | \
        sed 's/^\(KW.*\)-B /\1 /g' > $dirA/results 

      [ ! -f $datasetB/kws/utter_id ] && echo "File $datasetB/kws/utter_id must exist!" && exit 1;
      cat $kwsoutdir/result.* | \
        grep -F -f <(cut -f 1 -d ' ' $datasetB/kws/utter_id ) |\
        grep "^KW[-a-zA-Z0-9]*-A " | \
        sed 's/^\(KW.*\)-A /\1 /g' > $dirB/results
  done
fi

rootdirA=$decodedir/`basename $datasetA`
rootdirB=$decodedir/`basename $datasetB`
rootdirAB=$decodedir/`basename $datasetA`_`basename $datasetB`
rootdirBA=$decodedir/`basename $datasetB`_`basename $datasetA`


echo "Processing $datasetA"
if [ $stage -le 2 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirA/kws/kws_write_normalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirA/kws_LMWT/results \| \
    utils/write_kwslist.pl --flen=0.01 --duration=$durationA \
      --segments=$datadir/segments --normalize=true --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map  --digits=3 - $rootdirA/kws_LMWT/kwslist.xml || exit 1

  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirAB/kws/kws_write_normalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirAB/kws_LMWT/results \| \
    utils/write_kwslist.pl --flen=0.01 --duration=$durationA \
      --segments=$datadir/segments --normalize=true --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map  --digits=3 - $rootdirAB/kws_LMWT/kwslist.xml || exit 1
fi

if [ $stage -le 3 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirA/kws/kws_write_unnormalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirA/kws_LMWT/results \| \
    utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$durationA \
      --segments=$datadir/segments --normalize=false --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirA/kws_LMWT/kwslist.unnormalized.xml || exit 1
  
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirAB/kws/kws_write_unnormalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirAB/kws_LMWT/results \| \
    utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$durationA \
      --segments=$datadir/segments --normalize=false --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirAB/kws_LMWT/kwslist.unnormalized.xml || exit 1
fi

echo "Scoring $datasetA"
if [ $stage -le 4 ] ; then
  if [[ (! -x local/kws_score.sh ) ||  ($skip_scoring == true) ]] ; then
      echo "Not scoring, because the file local/kws_score.sh is not present" 
      exit 1
  elif [ ! -f $datasetA/kws/rttm ] ; then
      echo "Not scoring, because the file $datasetA/kws/rttm is not present"
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $rootdirA/kws/kws_scoring.LMWT.log \
      local/kws_score.sh $datasetA $rootdirA/kws_LMWT 
    $cmd LMWT=$min_lmwt:$max_lmwt $rootdirAB/kws/kws_scoring.LMWT.log \
      local/kws_score.sh --kwlist $datasetB/kws/kwlist.xml $datasetA $rootdirAB/kws_LMWT 
  fi
fi

echo "Processing $datasetB"
if [ $stage -le 5 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirB/kws/kws_write_normalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirB/kws_LMWT/results \| \
    utils/write_kwslist.pl --flen=0.01 --duration=$durationB \
      --segments=$datadir/segments --normalize=true --digits=3  --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirB/kws_LMWT/kwslist.xml || exit 1
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirBA/kws/kws_write_normalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirBA/kws_LMWT/results \| \
    utils/write_kwslist.pl --flen=0.01 --duration=$durationB \
      --segments=$datadir/segments --normalize=true --digits=3  --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirBA/kws_LMWT/kwslist.xml || exit 1
fi

if [ $stage -le 6 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirB/kws/kws_write_unnormalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirB/kws_LMWT/results \| \
    utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$durationB \
      --segments=$datadir/segments --normalize=false --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirB/kws_LMWT/kwslist.unnormalized.xml || exit 1
  $cmd LMWT=$min_lmwt:$max_lmwt $rootdirBA/kws/kws_write_unnormalized.LMWT.log \
    set -e';' set -o pipefail';' \
    cat $rootdirBA/kws_LMWT/results \| \
    utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$durationB \
      --segments=$datadir/segments --normalize=false --remove-dup=true\
      --map-utter=$kwsdatadir/utter_map - $rootdirBA/kws_LMWT/kwslist.unnormalized.xml || exit 1
fi

echo "Scoring $datasetB"
if [ $stage -le 7 ] ; then
  if [[ (! -x local/kws_score.sh ) ||  ($skip_scoring == true) ]] ; then
      echo "Not scoring, because the file local/kws_score.sh is not present"
  elif [ ! -f $datasetB/kws/rttm ] ; then
      echo "Not scoring, because the file $datasetB/kws/rttm is not present"
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $rootdirB/kws/kws_scoring.LMWT.log \
      local/kws_score.sh $datasetB $rootdirB/kws_LMWT || exit 1
    $cmd LMWT=$min_lmwt:$max_lmwt $rootdirBA/kws/kws_scoring.LMWT.log \
      local/kws_score.sh --kwlist $datasetA/kws/kwlist.xml $datasetB $rootdirBA/kws_LMWT || exit 1
  fi
fi

echo "Done, everything seems fine"
exit 0
