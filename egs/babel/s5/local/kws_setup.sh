#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.  
cmd=run.pl
case_insensitive=true
subset_ecf=
rttm_file=
extraid=
silence_word=  # Optional silence word to insert (once) between words of the transcript.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

help_message="$0: Initialize and setup the KWS task directory
Usage:
       $0  <ecf_file> <kwlist-file> [rttm-file] <lang-dir> <data-dir>
allowed switches:
      --subset-ecf /path/to/filelist     # The script will subset the ecf file 
                                         # to contain only the files from the filelist
      --rttm-file /path/to/rttm          # the preferred way how to specify the rttm
                                         # the older way (as an in-line parameter is 
                                         # obsolete and will be removed in near future
      --case-sensitive <true|false>      # Shall we be case-sensitive or not?
                                         # Please not the case-sensitivness depends 
                                         # on the shell locale!
              "

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ "$#" -ne "5" ] &&  [ "$#" -ne "4" ] ; then
    printf "FATAL: invalid number of arguments.\n\n"
    printf "$help_message\n"
    exit 1
fi

ecf_file=$1
kwlist_file=$2
if [ "$#" -eq "5" ] ; then
    rttm_file=$3
    langdir=$4
    datadir=$5
else
    langdir=$3
    datadir=$4
fi

# don't quote rttm_file as it's valid for it to be empty.
for filename in "$ecf_file" "$kwlist_file" $rttm_file; do
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

if [ ! -z $extraid ]; then
  kwsdatadir=$datadir/kws${extraid}
else
  kwsdatadir=$datadir/kws
fi

mkdir -p $kwsdatadir

if [ -z $subset_ecf ] ; then
  cp "$ecf_file" $kwsdatadir/ecf.xml || exit 1
else
  local/make_ecf_subset.sh $subset_ecf $ecf_file > $kwsdatadir/ecf.xml
fi

cp "$kwlist_file" $kwsdatadir/kwlist.xml || exit 1

if [ ! -z $rttm_file ] ; then
  cp "$rttm_file" $kwsdatadir/rttm || exit 1
fi

[ ! -z $silence_word ] && sil_opt="--silence-word $silence_word"
local/kws_data_prep.sh $sil_opt --case-insensitive $case_insensitive $langdir $datadir $kwsdatadir || exit 1

