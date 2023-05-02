#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

# Begin configuration section.
cmd=run.pl
duptime=0.5
model=final.mdl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage $0 [options] <lang-dir> <data-dir> <decode-dir>"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo ""
  exit 1;
fi

lang=$1;
data=$2;
decodedir=$3;


kwsdatadir=$data/kws
oracledir=$decodedir/kws_oracle
mkdir -p $oracledir/log

for filename in $lang/words.txt $decodedir/num_jobs \
                $data/text $decodedir/lat.1.gz \
                $decodedir/../$model ; do
    if [[ ! -f $filename ]] ; then
        echo "FATAL: File $filename does not exist!"
        exit 1;
    fi
done

nj=`cat $decodedir/num_jobs`

(cd $decodedir; ln -s ../$model final.mdl )
(cd $oracledir; echo "$nj" > num_jobs )

$cmd LAT=1:$nj $oracledir/log/lat.LAT.log \
  cat $data/text \| \
  sed 's/- / /g' \| \
  sym2int.pl --map-oov '"<unk>"' -f 2- $lang/words.txt \| \
  lattice-oracle --word-symbol-table=$lang/words.txt \
    --write-lattices="ark:|gzip -c > $oracledir/lat.LAT.gz" \
    "ark:gzip -cdf $decodedir/lat.LAT.gz|" ark:- ark,t:$oracledir/lat.LAT.tra;

