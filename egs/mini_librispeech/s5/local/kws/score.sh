#!/bin/bash

# Copyright 2012-2018  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
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
  if [ -z "$ntrue_from" ]; then
    for LMWT in $(seq $min_lmwt $max_lmwt) ; do
      mkdir -p ${kwsoutputdir}_$LMWT/details/
      mkdir -p ${kwsoutputdir}_$LMWT/scoring/

      # as we need to sweep through different ntrue-scales we will
      # we will do it in one parallel command -- it will be more effective
      # than sweeping in a loop and for all lmwts in parallel (as usuallyu
      # there will be just a couple of different lmwts, but the ntrue-scale
      # has a larger dynamic range
      $cmd NTRUE=1:21 $kwsoutputdir/log/score.${LMWT}.NTRUE.log \
        ntrue=\$\(perl -e 'print 1+(NTRUE-1)/5.0' \) '&&' \
        cat ${kwsoutputdir}_$LMWT/results \|\
          local/kws/normalize_results_kst.pl --trials $trials --ntrue-scale \$ntrue \|\
          local/kws/filter_kws_results.pl --probs --nbest 200   \|\
          compute-atwv $trials ark,t:$kwsdatadir/hitlist ark:- \
          \> ${kwsoutputdir}_$LMWT/scoring/score.NTRUE.txt

      ntrue=$(grep ATWV ${kwsoutputdir}_$LMWT/scoring/score.*.txt | \
              sort -k2,2nr -t '='  | head -n 1 | \
              sed 's/.*score\.\([0-9][0-9]*\)\.txt.*/\1/g')
      #The calculation of ntrue must be the same as in the command above
      echo "$ntrue" > ${kwsoutputdir}_$LMWT/details/ntrue_raw
      ntrue=$(perl -e "print 1+($ntrue-1)/5.0")
      echo "$ntrue" > ${kwsoutputdir}_$LMWT/details/ntrue
    done
  else
    for LMWT in $(seq $min_lmwt $max_lmwt) ; do
      mkdir -p ${kwsoutputdir}_$LMWT/details/
      mkdir -p ${kwsoutputdir}_$LMWT/scoring/

      cp ${ntrue_from}_${LMWT}/details/ntrue  ${kwsoutputdir}_${LMWT}/details/ntrue
      [ -f  ${ntrue_from}_${LMWT}/details/ntrue_raw ] && \
        cp ${ntrue_from}_${LMWT}/details/ntrue_raw  ${kwsoutputdir}_${LMWT}/details/ntrue_raw
      echo "$ntrue_from" > ${kwsoutputdir}_${LMWT}/details/ntrue_from
    done
  fi
fi

if [ $stage -le 1 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/normalize.LMWT.log \
    cat ${kwsoutputdir}_LMWT/results \|\
      local/kws/normalize_results_kst.pl --trials $trials --ntrue-scale \$\(cat ${kwsoutputdir}_LMWT/details/ntrue\)\
      \> ${kwsoutputdir}_LMWT/details/results

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/score.final.LMWT.log \
    cat ${kwsoutputdir}_LMWT/details/results \|\
      compute-atwv $trials ark,t:$kwsdatadir/hitlist ark:- \
      ${kwsoutputdir}_LMWT/details/alignment.csv \> ${kwsoutputdir}_LMWT/details/score.txt  '&&' \
    cp ${kwsoutputdir}_LMWT/details/score.txt ${kwsoutputdir}_LMWT/score.txt

  if [ -f $kwsdatadir/categories ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/per-category-stats.LMWT.log \
      cat ${kwsoutputdir}_LMWT/details/alignment.csv \|\
        perl local/search/per_category_stats.pl --sweep-step 0.005  $trials \
        $kwsdatadir/categories \> ${kwsoutputdir}_LMWT/details/per-category-score.txt
  else
    echo "$0: Categories file not found, not generating per-category scores"
  fi
fi

if [ $stage -le 2 ]; then
if [ -f $kwsdatadir/f4de_attribs ] ; then
  language=""
  flen=0.01
  kwlist_name=""
  . $kwsdatadir/f4de_attribs #override the previous variables

  ecf=$kwsdatadir/ecf.xml
  rttm=$kwsdatadir/rttm
  kwlist=$kwsdatadir/kwlist.xml

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/f4de_prepare.LMWT.log \
    mkdir -p ${kwsoutputdir}_LMWT/f4de/ '&&' cat $kwlist \| \
    local/search/annotate_kwlist.pl $kwsdatadir/categories \> ${kwsoutputdir}_LMWT/f4de/kwlist.xml

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/f4de_write_kwslist.LMWT.log \
    cat ${kwsoutputdir}_LMWT/details/results \| \
      utils/int2sym.pl -f 2 $kwsdatadir/utt.map \| \
      local/search/utt_to_files.pl --flen $flen $kwsdatadir/../segments \|\
      local/search/write_kwslist.pl --flen $flen --language $language \
      --kwlist-id $kwlist_name \> ${kwsoutputdir}_LMWT/f4de/kwslist.xml

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/f4de_score.LMWT.log \
    KWSEval -e $ecf -r $rttm -t ${kwsoutputdir}_LMWT/f4de/kwlist.xml -a  \
      --zGlobalMeasures Optimum --zGlobalMeasures Supremum \
      -O -B -q 'Characters:regex=.*' -q 'NGramOrder:regex=.*' \
      -O -B -q 'OOV:regex=.*' -q 'BaseOOV:regex=.*' \
      -s ${kwsoutputdir}_LMWT/f4de/kwslist.xml -c -o -b -d -f  ${kwsoutputdir}_LMWT/f4de/

  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutputdir/log/f4de_report.LMWT.log \
    local/kws_oracle_threshold.pl --duration $trials \
      ${kwsoutputdir}_LMWT/f4de/alignment.csv \> ${kwsoutputdir}_LMWT/f4de/metrics.txt
fi
fi

echo "$0: Done"
exit 0;


