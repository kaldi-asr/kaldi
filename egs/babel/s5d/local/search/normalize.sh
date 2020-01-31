#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.
# case_insensitive=true
extraid=
min_lmwt=8
max_lmwt=12
cmd=run.pl
stage=0
ntrue_from=
# End configuration section.

help_message="$0: score the kwslist using the F4DE scorer from NIST
  Example:
    $0 [additional-parameters] <kaldi-data-dir> <kws-results-dir>
    where the most important additional parameters can be:
    --extraid  <extra-id> #for using, when a non-default kws tasks are setup
              (using the kws_setup.sh --extraid) for a kaldi-single data-dir"

echo $0 $@
[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
    printf "FATAL: incorrect number of variables given to the script\n\n"
    printf "$help_message\n"
    exit 1;
fi

set -e -o pipefail

langdir=$1
if [ -z $extraid ] ; then
  kwsdatadir=$2/kws
else
  kwsdatadir=$2/kwset_${extraid}
fi
kwsoutputdir="$3"

trials=$(cat $kwsdatadir/trials)
mkdir -p $kwsoutputdir/log/

if [ $stage -le 0 ] ; then
  for LMWT in $(seq $min_lmwt $max_lmwt) ; do
    mkdir -p ${kwsoutputdir}_$LMWT/details/

    cp ${ntrue_from}_$LMWT/details/ntrue  ${kwsoutputdir}_$LMWT/details/ntrue
    cp ${ntrue_from}_$LMWT/details/ntrue_raw  ${kwsoutputdir}_$LMWT/details/ntrue_raw
    echo "$ntrue_from" > ${kwsoutputdir}_$LMWT/details/ntrue_from
  done
fi

if [ $stage -le 1 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/normalize.LMWT.log \
    cat ${kwsoutputdir}_LMWT/results \|\
      local/search/normalize_results_kst.pl --trials $trials --ntrue-scale \$\(cat ${kwsoutputdir}_LMWT/details/ntrue\)\
      \> ${kwsoutputdir}_LMWT/details/results

fi

if [ $stage -le 2 ]; then
if [ -f $kwsdatadir/f4de_attribs ] ; then
  language=""
  flen=0.01
  kwlist_name=""
  . $kwsdatadir/f4de_attribs #override the previous variables

  ecf=$kwsdatadir/ecf.xml
  kwlist=$kwsdatadir/kwlist.xml

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/f4de_write_kwslist.LMWT.log \
    mkdir -p ${kwsoutputdir}_LMWT/f4de/\; \
    cat ${kwsoutputdir}_LMWT/details/results \| \
      utils/int2sym.pl -f 2 $kwsdatadir/utt.map \| \
      local/search/utt_to_files.pl --flen $flen $kwsdatadir/../segments \|\
      local/search/write_kwslist.pl --flen $flen --language $language \
      --kwlist-id $kwlist_name \> ${kwsoutputdir}_LMWT/f4de/kwslist.xml

fi
fi

echo "$0: Done"
exit 0;


