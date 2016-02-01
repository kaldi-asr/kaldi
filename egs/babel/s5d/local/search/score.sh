#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.
# case_insensitive=true
extraid=
min_lmwt=8
max_lmwt=12
cmd=run.pl
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

langdir=$1
if [ -z $extraid ] ; then
  kwsdatadir=$2/kws
else
  kwsdatadir=$2/kwset_${extraid}
fi
kwsoutputdir="$3"

trials=$(cat $kwsdatadir/trials)
mkdir -p $kwsoutputdir/log/

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
      local/search/normalize_results_kst.pl --ntrue-scale \$ntrue \|\
      local/search/filter_kws_results.pl --nbest 200   \|\
      compute-atwv $trials ark,t:$kwsdatadir/hitlist ark:- \
      \> ${kwsoutputdir}_$LMWT/scoring/score.NTRUE.txt

  ntrue=$(grep ATWV ${kwsoutputdir}_$LMWT/scoring/score.*.txt | \
          sort -k2,2nr -t '='  | head -n 1 | \
          sed 's/.*score\.\([0-9][0-9]*\)\.txt.*/\1/g')
  #The calculation of ntrue must be the same as in the command above
  ntrue=$(perl -e "print 1+($ntrue-1)/5.0")
  echo "$ntrue" > ${kwsoutputdir}_$LMWT/details/ntrue

  cat ${kwsoutputdir}_$LMWT/results |\
    local/search/normalize_results_kst.pl --ntrue-scale $ntrue \
    > ${kwsoutputdir}_$LMWT/details/results \
    2> ${kwsoutputdir}_$LMWT/log/normalize.log

  cat ${kwsoutputdir}_$LMWT/details/results |\
    compute-atwv $trials ark,t:$kwsdatadir/hitlist ark:- \
      ${kwsoutputdir}_$LMWT/details/alignment.csv \
       > ${kwsoutputdir}_$LMWT/details/score.txt \
       2> ${kwsoutputdir}_$LMWT/log/score.log

  cat ${kwsoutputdir}_$LMWT/details/alignment.csv |\
    perl local/search/per_category_stats.pl --sweep-step 0.005  $trials\
    $kwsdatadir/categories \
      > ${kwsoutputdir}_$LMWT/details/per-category-score.txt \
      2> ${kwsoutputdir}_$LMWT/log/per-category-score.txt
  
  cp ${kwsoutputdir}_$LMWT/details/score.txt ${kwsoutputdir}_$LMWT/score.txt
done

echo "$0: Done"
exit 0;


