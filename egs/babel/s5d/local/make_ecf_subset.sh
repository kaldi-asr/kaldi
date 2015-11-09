#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.

echo "$0 $@" 1>&2 # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

help_message="$0: generates an subset ecf file for spoken term detection evaluation. 
                The first parameter specifies the descriptor of the subset,
                the second parameter specifies the original ecf file.
                The file will be generated in the kws subdirectory of the directory
                given as a third parameter and will be named ecf.xml
                Output goes to stdout.
             Usage:
                 $0 <subset-descriptor-list-file> <source-ecf-file> "


[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ "$#" -ne "2" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n" 1>&2
    printf "$help_message\n\n" 1>&2
    exit 1;
fi

list_file=$1
src_ecf_file=$2

if [[ ! -f "$list_file" ]]; then
    echo -e "FATAL: The list file does not exist! \n\n" 1>&2
    printf "$help_message\n" 1>&2
    exit 1;
fi
if [[ ! -f "$src_ecf_file" ]]; then
    echo -e "FATAL: The source ecf file does not exist! \n\n" 1>&2
    printf "$help_message\n" 1>&2
    exit -1
fi


duration=`grep -F -f $list_file $src_ecf_file | sed "s/.*dur=\"\([0-9.][0-9.]*\).*/\1  /g" | awk '{x += $1;} END{print x;}'`

# Output is produced here:
(
  grep "<ecf" $src_ecf_file | sed "s/source_signal_duration=\\\"[0-9.][0-9.]*\\\"/source_signal_duration=\"$duration\"/g" | head -n 1;
  grep -F -f $list_file $src_ecf_file 
  echo "</ecf>"
)
