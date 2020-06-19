#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0

# Begin configuration section.  
cmd=run.pl
nbest=-1
strict=true
indices_dir=
frame_subsampling_factor=1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: steps/search_index.sh [options] <kws-data-dir> <kws-dir>"
   echo " e.g.: steps/search_index.sh data/kws exp/sgmm2_5a_mmi/decode/kws/"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --nbest <int>                                    # return n best results. (-1 means all)"
   echo "  --indices-dir <path>                             # where the indices should be stored, by default it will be in <kws-dir>"
   exit 1;
fi


kwsdatadir=$1;
kwsdir=$2;

if [ -z $indices_dir ] ; then
  indices_dir=$kwsdir
fi

mkdir -p $kwsdir/log;
nj=`cat $indices_dir/num_jobs` || exit 1;
if [ -f $kwsdatadir/keywords.fsts.gz ]; then
  keywords="\"gunzip -c $kwsdatadir/keywords.fsts.gz|\""
elif [ -f $kwsdatadir/keywords.fsts ]; then
  keywords=$kwsdatadir/keywords.fsts;
else
  echo "$0: no such file $kwsdatadir/keywords.fsts[.gz]" && exit 1;
fi

for f in $indices_dir/index.1.gz ; do
  [ ! -f $f ] && echo "make_index.sh: no such file $f" && exit 1;
done

$cmd JOB=1:$nj $kwsdir/log/search.JOB.log \
  kws-search --strict=$strict --negative-tolerance=-1 \
  --frame-subsampling-factor=${frame_subsampling_factor} \
  "ark:gzip -cdf $indices_dir/index.JOB.gz|" ark:$keywords \
  "ark,t:|gzip -c > $kwsdir/result.JOB.gz" \
  "ark,t:|gzip -c > $kwsdir/stats.JOB.gz" || exit 1;

exit 0;
