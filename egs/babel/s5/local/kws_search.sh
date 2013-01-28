#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


help_message="$0: create subset of the input directory (specified as the first directory).
                 The subset is specified by the second parameter.
                 The directory in which the subset should be created is the third parameter
             Example:
                 $0 <source-corpus-dir> <subset-descriptor-list-file> <target-corpus-subset-dir>"

# Begin configuration section.  
#acwt=0.0909091
min_lmwt=7
max_lmwt=17
duptime=0.6
cmd=run.pl
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [[ "$#" -ne "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi


datadir=$2
langdir=$1
decodedir=$3

kwsdatadir=$datadir/kws

if [ ! -d "$datadir"  ] || [ ! -d "$kwsdatadir" ] ; then
    echo "FATAL: the data directory does not exist"
    exit 1;
if
if [[ ! -d "$langdir"  ]] ; then
    echo "FATAL: the lang directory does not exist"
    exit 1;
if
if [[ ! -d "$decodedir"  ]] ; then
    echo "FATAL: the directory with decoded files does not exist"
    exit 1;
if
if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
    echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
    exit 1;
if


duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

for lmwt in `seq $min_lmwt $max_lmwt` ; do
    kwsoutdir=$decodedir/kws_$lmwt
    mkdir -p $kwsoutdir

    acwt=`echo "scale=5; 1/$lmwt" | bc -l | sed "s/^./0./g"` 
    local/make_index.sh --cmd "$cmd" --acwt $acwt \
      $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1

    local/search_index.sh $kwsdatadir $kwsoutdir  || exit 1

    cat $kwsoutdir/result.* | \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map \
        - - | \
      local/filter_kwslist.pl $duptime > $kwsoutdir/kwslist.xml

done

