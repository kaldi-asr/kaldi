#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


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
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

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
    if [ ! -d "$splitdatadir"  ] || [ ! -d "$kwsdatadir" ] ; then
        echo "FATAL: the data directory does not exist"
        exit 1;
    fi
    if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
        echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
        exit 1;
    fi
done

kwsdatadir=$datadir/kws

durationA=`head -1 $datasetA/kws/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

durationB=`head -1 $datasetB/kws/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

if [ ! -z "$model" ]; then
    model_flags="--model $model"
fi

for lmwt in `seq $min_lmwt $max_lmwt` ; do
    kwsoutdir=$decodedir/kws_$lmwt
    dirA=$decodedir/`basename $datasetA`/kws_$lmwt
    dirB=$decodedir/`basename $datasetB`/kws_$lmwt
    mkdir -p $kwsoutdir
    mkdir -p $dirA
    mkdir -p $dirB

    acwt=`echo "scale=5; 1/$lmwt" | bc -l | sed "s/^./0./g"` 
    local/make_index.sh --cmd "$cmd" --acwt $acwt $model_flags\
      $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1

    local/search_index.sh --cmd "$cmd" $kwsdatadir $kwsoutdir  || exit 1

    cat $kwsoutdir/result.* | \
      grep -F -f <(cut -f 1 -d ' ' $datasetA/kws/utter_id ) |\
      grep "^KW[-a-zA-Z0-9]*-A " | \
      sed 's/^\(KW.*\)-A /\1 /g' > $dirA/results

    cat $dirA/results | \
      utils/write_kwslist.pl --flen=0.01 --duration=$durationA \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $dirA/kwslist.xml
   
    cat $dirA/results | \
      utils/write_kwslist.pl --flen=0.01 --duration=$durationA \
        --segments=$datadir/segments --normalize=false \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $dirA/kwslist.unnormalized.xml

    if [[ (! -x local/kws_score.sh ) ||  ($skip_scoring == true) ]] ; then
        echo "Not scoring, because the file local/kws_score.sh is not present"
    else
        local/kws_score.sh $datasetA $dirA 
    fi

    cat $kwsoutdir/result.* | \
      grep -F -f <(cut -f 1 -d ' ' $datasetB/kws/utter_id ) |\
      grep "^KW[-a-zA-Z0-9]*-B " | \
      sed 's/^\(KW.*\)-B /\1 /g' > $dirB/results

    cat $dirB/results | \
      utils/write_kwslist.pl --flen=0.01 --duration=$durationB \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $dirB/kwslist.xml
   
    cat $dirB/results | \
      utils/write_kwslist.pl --flen=0.01 --duration=$durationB \
        --segments=$datadir/segments --normalize=false \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $dirB/kwslist.unnormalized.xml

    if [[ (! -x local/kws_score.sh ) ||  ($skip_scoring == true) ]] ; then
        echo "Not scoring, because the file local/kws_score.sh is not present"
    else
        local/kws_score.sh $datasetB $dirB 
    fi
done

