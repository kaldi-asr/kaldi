#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

. ./path.sh
. ./cmd.sh

# Begin configuration section.  
acwt=0.0909091
duptime=0.6
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

datadir=$2
langdir=$1
decodedir=$3

kwsdatadir=$datadir/kws
kwsoutdir=$decodedir/kws

mkdir -p $kwsdatadir
mkdir -p $kwsoutdir


local/make_index.sh --cmd "$cmd" --acwt $acwt \
  $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1

local/search_index.sh $kwsdatadir $kwsoutdir  || exit 1

duration=`head -1 $kwsdatadir/ecf.xml |\
  grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
  grep -o -E "[0-9]*[\.]*[0-9]*" |\
  perl -e 'while(<>) {print $_/2;}'`

cat $kwsoutdir/result.* | \
    utils/write_kwslist.pl --flen=0.01 --duration=$duration \
    --segments=$datadir/segments --normalize=true \
    --map-utter=$kwsdatadir/utter_map \
    - - | \
    utils/filter_kwslist.pl $duptime > $kwsoutdir/kwslist.xml

