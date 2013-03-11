#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


help_message="$0: create subset of the input directory (specified as the first directory).
                 The subset is specified by the second parameter.
                 The directory in which the subset should be created is the third parameter
             Example:
                 $0 <lang-dir> <data-dir> <decode-dir>"

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

if [[ "$#" -ne "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi


langdir=$1
datadir=$2
decodedir=$3

kwsdatadir=$datadir/kws

if [ ! -d "$datadir"  ] || [ ! -d "$kwsdatadir" ] ; then
    echo "FATAL: the data directory does not exist"
    exit 1;
fi
if [[ ! -d "$langdir"  ]] ; then
    echo "FATAL: the lang directory does not exist"
    exit 1;
fi
if [[ ! -d "$decodedir"  ]] ; then
    echo "FATAL: the directory with decoded files does not exist"
    exit 1;
fi
if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
    echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
    exit 1;
fi


duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

if [ ! -z "$model" ]; then
    model_flags="--model $model"
else
    model_flags=
fi

for lmwt in `seq $min_lmwt $max_lmwt` ; do
    kwsoutdir=$decodedir/kws_$lmwt
    mkdir -p $kwsoutdir

    acwt=`echo "scale=5; 1/$lmwt" | bc -l | sed "s/^./0./g"` 
    local/make_index.sh --cmd "$cmd" --acwt $acwt $model_flags\
      $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1

    local/search_index.sh --cmd "$cmd" $kwsdatadir $kwsoutdir  || exit 1

    cat $kwsoutdir/result.* | \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $kwsoutdir/kwslist.xml
   
    cat $kwsoutdir/result.* | \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$datadir/segments --normalize=false \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $kwsoutdir/kwslist.unnormalized.xml

    if [[ (! -x local/kws_score.sh ) ||  ($skip_scoring == true) ]] ; then
        echo "Not scoring, because the file local/kws_score.sh is not present"
    else
        local/kws_score.sh $datadir $kwsoutdir
    fi
done

