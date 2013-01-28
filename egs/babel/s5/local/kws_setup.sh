#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.  
cmd=run.pl
case_insensitive=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

help_message="$0: Initialize and setup the KWS task directory
Example:
              $0 <ecf_file> <kwlist-file> <rttm-file> <lang-dir> <data-dir>"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ "$#" -ne "5" ]] ; then
    printf "FATAL: invalid number of arguments.\n\n"
    printf "$help_message\n"
    exit 1
fi

ecf_file=$1
kwlist_file=$2
rttm_file=$3
langdir=$4
datadir=$5

for filename in "$ecf_file" "$kwlist_file" "$rttm_file" ; do
    echo $filename
    if [ ! -f $filename ] ; then
        printf "FATAL: filename \'$filename\' does not refer to a valid file\n"
        printf "$help_message\n"
        exit 1;
    fi
done
for dirname in "$langdir" "$datadir" ; do
    if [ ! -d $dirname ] ; then
        printf "FATAL: dirname \'$dirname\' does not refer to a valid directory\n"
        printf "$help_message\n"
        exit 1;
    fi
done


kwsdatadir=$datadir/kws
mkdir -p $kwsdatadir

cp `readlink -f $ecf_file` $kwsdatadir/ecf.xml || exit 1
cp `readlink -f $kwlist_file` $kwsdatadir/kwlist.xml || exit 1
cp `readlink -f $rttm_file` $kwsdatadir/rttm || exit 1

local/kws_data_prep.sh --case-insensitive $case_insensitive $langdir $datadir $kwsdatadir || exit 1

