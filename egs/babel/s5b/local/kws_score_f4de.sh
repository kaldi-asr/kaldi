#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.
# case_insensitive=true
extraid=
kwlist=
f4de_prefix=
# End configuration section.

help_message="$0: score the kwslist using the F4DE scorer from NIST
  Example:
    $0 [additional-parameters] <kaldi-data-dir> <kws-results-dir>
    where the most important additional parameters can be:
    --extraid  <extra-id> #for using, when a non-default kws tasks are setup 
              (using the kws_setup.sh --extraid) for a kaldi-single data-dir
    --kwlist <kwlist> #allows for an alternative kwlist -- if not set, the default
              kwlist is taken from <kaldi-data-dir>
    --f4de-prefix <prefix-id> #allows for scoring the same results using 
              different kwlists and storing them in the same dir "

echo $0 $@
[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
    printf "FATAL: incorrect number of variables given to the script\n\n"
    printf "$help_message\n"
    exit 1;
fi

if [ -z $extraid ] ; then
  kwsdatadir=$1/kws
else
  kwsdatadir=$1/${extraid}_kws
fi
kwsoutputdir="$2/"

if [ -z $kwlist ] ; then
  kwlist=$kwsdatadir/kwlist.xml
fi

if [ ! -z ${f4de_prefix} ] ; then
  f4de_prefix="/${f4de_prefix}"
fi

if [[ ! -d "$kwsdatadir" ]] ; then
    echo "FATAL: the KWS input data directory does not exist!"
    exit 1;
fi

for file in $kwsdatadir/ecf.xml $kwsdatadir/rttm $kwlist ; do
    if [[ ! -f "$file" ]] ; then
        echo "FATAL: file $file does not exist!"
        exit 1;
    fi
done

echo KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t $kwlist \
    -s $kwsoutputdir/kwslist.xml -c -o -b -d -f $kwsoutputdir

KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t $kwlist \
    -s $kwsoutputdir/kwslist.xml -c -o -b -d -f ${kwsoutputdir}${f4de_prefix} || exit 1;

duration=`cat ${kwsoutputdir}${f4de_prefix}/sum.txt | grep TotDur | cut -f 3 -d '|' | sed "s/\s*//g"`

local/kws_oracle_threshold.pl --duration $duration ${kwsoutputdir}${f4de_prefix}/alignment.csv > ${kwsoutputdir}${f4de_prefix}/metrics.txt

exit 0;


