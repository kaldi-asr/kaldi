#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


help_message="$(basename $0): do keyword indexing and search.  data-dir is assumed to have
                 kws/ subdirectory that specifies the terms to search for.  Output is in
                 decode-dir/kws/
             Usage:
                 $(basename $0) <lang-dir> <data-dir> <decode-dir>"

# Begin configuration section.  
#acwt=0.0909091
min_lmwt=7
max_lmwt=17
duptime=0.6
cmd=run.pl
model=
skip_scoring=false
skip_optimization=false # Should never be necessary to specify true here.
max_states=150000
stage=0
word_ins_penalty=0
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [[ "$#" -ne "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi


langdir=$1
datadir=$2
decodedir=$3

kwsdatadir=$datadir/kws

if [ ! -d "$datadir"  ] || [ ! -d "$kwsdatadir" ] ; then
    echo "FATAL: the data directory does not exist"
    exit 1;
fi
if [[ ! -d "$langdir"  ]] ; then
    echo "FATAL: the lang directory does not exist"
    exit 1;
fi
if [[ ! -d "$decodedir"  ]] ; then
    echo "FATAL: the directory with decoded files does not exist"
    exit 1;
fi
if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
    echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
    exit 1;
fi


duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

if [ ! -z "$model" ]; then
    model_flags="--model $model"
else
    model_flags=
fi

if [ $stage -le 0 ] ; then
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      kwsoutdir=$decodedir/kws_$lmwt
      mkdir -p $kwsoutdir

      acwt=`echo "scale=5; 1/$lmwt" | bc -l | sed "s/^./0./g"` 
      steps/make_index.sh --cmd "$cmd" --acwt $acwt $model_flags\
        --skip-optimization $skip_optimization --max-states $max_states \
        --word-ins-penalty $word_ins_penalty \
        $kwsdatadir $langdir $decodedir $kwsoutdir  || exit 1
  done
fi

if [ $stage -le 1 ]; then
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      kwsoutdir=$decodedir/kws_$lmwt
      mkdir -p $kwsoutdir
      local/search_index.sh --cmd "$cmd" $kwsdatadir $kwsoutdir  || exit 1
  done
fi

if [ $stage -le 2 ]; then
  mkdir -p $decodedir/kws/
  echo "Writing normalized results"
  $cmd LMWT=$min_lmwt:$max_lmwt $decodedir/kws/kws_write_normalized.LMWT.log \
    set -e ';' set -o pipefail ';'\
    cat $decodedir/kws_LMWT/result.* \| \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map --digits=3 \
        - - \| local/filter_kwslist.pl $duptime '>' $decodedir/kws_LMWT/kwslist.xml || exit 1
fi

if [ $stage -le 3 ]; then
  echo "Writing unnormalized results"
  $cmd LMWT=$min_lmwt:$max_lmwt $decodedir/kws/kws_write_unnormalized.LMWT.log \
    set -e ';' set -o pipefail ';'\
    cat $decodedir/kws_LMWT/result.* \| \
        utils/write_kwslist.pl --flen=0.01 --duration=$duration \
          --segments=$datadir/segments --normalize=false \
          --map-utter=$kwsdatadir/utter_map \
          - - \| local/filter_kwslist.pl $duptime '>' $decodedir/kws_LMWT/kwslist.unnormalized.xml || exit 1;
fi

if [ $stage -le 4 ]; then
  if [[ (! -x local/kws_score.sh ) ]] ; then
    echo "Not scoring, because the file local/kws_score.sh is not present"
  elif [[ $skip_scoring == true ]] ; then
    echo "Not scoring, because --skip-scoring true was issued"
  else
    echo "Scoring KWS results"
    $cmd LMWT=$min_lmwt:$max_lmwt $decodedir/kws/kws_scoring.LMWT.log \
       local/kws_score.sh $datadir $decodedir/kws_LMWT || exit 1;
  fi
fi

exit 0
