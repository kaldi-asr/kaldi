#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Jan Trmal)
#           2013  Johns Hopkins University 
# Apache 2.0.

. ./path.sh
. ./cmd.sh

# Begin configuration section.  
cmd=run.pl
acwt=0.09091 #Acoustic weight -- should not be necessary for oracle lattices 
duptime=0.6  #Max time difference in which the occurences of the same KW will be seen as duplicates
text=     # an alternative reference text to use. when not specified, the <data-dir>/text will be used
model=    # acoustic model to use 
extraid=  # kws setup extra ID (kws task was setup using kws_setup.sh --extraid <id>
stage=0   # to resume the computation from different stage
# End configuration section.

set -e 
set -o pipefail

echo "$0 $@"  # Print the command line for logging


[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage $0 [options] <lang-dir> <data-dir> <decode-dir>"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --text <text-file> #The alternative text file in the format SEGMENT W1 W2 W3..., "
  echo "                     #The default text file is taken from <data-dir>/text"
  echo ""
  exit 1;
fi

lang=$1;
data=$2;
decodedir=$3;

if [ -z $text ] ; then
  text=$data/text
fi

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  srcdir=`dirname $decodedir`; # The model directory is one level up from decoding directory.
  model=$srcdir/final.mdl; 
fi

if [ -z $extraid ] ; then # the same logic as with kws_setup.sh
  kwsdatadir=$data/kws
else
  kwsdatadir=$data/${extraid}_kws
fi

nj=`cat $decodedir/num_jobs`;

oracledir=$decodedir/kws_oracle
mkdir -p $oracledir 
mkdir -p $oracledir/log

if [ $stage -le 0 ] ; then
  echo "$nj" > $oracledir/num_jobs
  $cmd LAT=1:$nj $oracledir/log/oracle_lat.LAT.log \
    cat $text \| \
    sed 's/- / /g' \| \
    sym2int.pl --map-oov '"<unk>"' -f 2- $lang/words.txt \| \
    lattice-oracle --word-symbol-table=$lang/words.txt \
      --write-lattices="ark:|gzip -c > $oracledir/lat.LAT.gz" \
      "ark:gzip -cdf $decodedir/lat.LAT.gz|" ark:- ark,t:$oracledir/lat.LAT.tra;
fi

if [ $stage -le 1 ] ; then
  steps/make_index.sh --cmd "$cmd" --acwt $acwt --model $model  \
    $kwsdatadir $lang $oracledir $oracledir
fi

if [ $stage -le 2 ] ; then
  steps/search_index.sh --cmd "$cmd" $kwsdatadir $oracledir
fi

if [ $stage -le 3 ]; then

  #TODO: this stage should be probably moved in a single script file
  # and used accross all the kw search scripts
  duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`


  cat $oracledir/result.* | \
    utils/write_kwslist.pl --flen=0.01 --duration=$duration \
    --segments=$data/segments --normalize=true --duptime=$duptime\
    --map-utter=$kwsdatadir/utter_map --remove-dup=true \
    -  $oracledir/kwslist_orig.xml

  #This does not do much -- just adds empty entries for keywords for which
  #not even one occurence has not been found
  local/fix_kwslist.pl $kwsdatadir/kwlist.xml $oracledir/kwslist_orig.xml $oracledir/kwslist.xml
fi


if [ $stage -le 4 ]; then
  #As there is a missing functionality in the F4DE for scoring
  #subsets of the original set, lets  keep this commented out.
  #Alternatively:TODO: write a filter_kwslist.pl script
  #That will produce kwslist on a basis of given kwlist.xml subset

  local/kws_score_f4de.sh `dirname $kwsdatadir` $oracledir
  #-local/kws_score_f4de.sh --kwlist $kwsdatadir/kwlist_outvocab.xml \
  #-  --f4de-prefix outvocab `dirname $kwsdatadir` $oracledir || exit 1
  #-local/kws_score_f4de.sh --kwlist $kwsdatadir/kwlist_invocab.xml \
  #-  --f4de-prefix invocab `dirname $kwsdatadir` $oracledir || exit 1

  echo "======================================================="
  (
    echo -n "ATWV-full     "
    grep Occurrence $oracledir/sum.txt | cut -d '|' -f 13  
  )

  #-(
  #-echo -n "ATWV-invocab  "
  #-grep Occurrence $oracledir/invocab.sum.txt | cut -d '|' -f 13  
  #-) || echo "Error occured getting the invocab results"

  #-(
  #-echo -n "ATWV-outvocab "
  #-grep Occurrence $oracledir/outvocab.sum.txt | cut -d '|' -f 13  
  #-) || echo "Error occured getting the outvocab results"

  echo "======================================================="
fi
