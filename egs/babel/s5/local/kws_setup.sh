#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

. ./path.sh
. ./cmd.sh

# Begin configuration section.  
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

ecf_file=$1
kwlist_file=$2
rttm_file=$3
langdir=$4
datadir=$5
kwsdatadir=$datadir/kws

mkdir -p $kwsdatadir

cp `readlink -f $ecf_file` $kwsdatadir/ecf.xml || exit 1
cp `readlink -f $kwlist_file` $kwsdatadir/kwlist.xml || exit 1
cp `readlink -f $rttm_file` $kwsdatadir/rttm || exit 1

local/kws_data_prep.sh --case-insensitive true $langdir $datadir $kwsdatadir || exit 1

