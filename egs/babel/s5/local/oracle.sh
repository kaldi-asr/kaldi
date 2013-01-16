#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

. ./path.sh
. ./cmd.sh
. /export/babel/data/env.sh
. /export/babel/data/software/env.sh


# Begin configuration section.  
cmd=run.pl
acwt=0.09091
duptime=0.5
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage $0 [options] <lang-dir> <data-dir> <kws-data-dir> <decode-dir>"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo ""
  exit 1;
fi

lang=$1;
data=$2;
kwsdatadir=$3
decodedir=$4;

nj=`cat $decodedir/num_jobs`;

oracledir=$decodedir/oracle-`basename $data`
mkdir -p $oracledir
mkdir -p $oracledir/log

echo "$nj" > $oracledir/num_jobs
$cmd LAT=1:$nj $oracledir/log/lat.LAT.log \
  cat $data/text \| \
  sed 's/- / /g' \| \
  sym2int.pl --map-oov '"<unk>"' -f 2- $lang/words.txt \| \
  lattice-oracle --word-symbol-table=$lang/words.txt \
    --write-lattices="ark:|gzip -c > $oracledir/lat.LAT.gz" \
    "ark:gzip -cdf $decodedir/lat.LAT.gz|" ark:- ark,t:$oracledir/lat.LAT.tra;

ln -s `readlink -f $oracledir/../../final.mdl` `readlink -f $oracledir/../final.mdl`
babel/make_index.sh --cmd "$cmd" --acwt $acwt \
  $kwsdatadir $lang $oracledir $oracledir/index  || exit 1

duration=`head -1 $kwsdatadir/ecf.xml |\
  grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
  grep -o -E "[0-9]*[\.]*[0-9]*" |\
  perl -e 'while(<>) {print $_/2;}'`

babel/search_index.sh $kwsdatadir $oracledir/index  || exit 1

cat $oracledir/index/result.* | \
    babel/write_kwslist.pl --flen=0.01 --duration=$duration \
    --segments=$data/segments --normalize=true \
    --map-utter=$kwsdatadir/utter_map \
    - - | \
    babel/filter_kwslist.pl $duptime > $oracledir/kwslist.xml || exit 1

KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t $kwsdatadir/kws.xml \
    -s $oracledir/kwslist.xml \
    -c -o -b -d -f $oracledir/kws

KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t babel/subsets/keyword_outvocab.xml \
    -s $oracledir/kwslist.xml \
    -c -o -b -d -f $oracledir/outvocab

KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t babel/subsets/keyword_invocab.xml \
    -s $oracledir/kwslist.xml \
    -c -o -b -d -f $oracledir/invocab

echo "======================================================="
echo -n "ATWV-full     "
grep Occurrence $oracledir/kws.sum.txt | cut -d '|' -f 13  
echo -n "ATWV-invocab  "
grep Occurrence $oracledir/invocab.sum.txt | cut -d '|' -f 13  
echo -n "ATWV-outvocab "
grep Occurrence $oracledir/outvocab.sum.txt | cut -d '|' -f 13  
echo "======================================================="

