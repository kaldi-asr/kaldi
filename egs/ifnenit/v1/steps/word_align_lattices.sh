#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey)  2012
# Apache 2.0.

# Begin configuration section.
silence_label=0
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

for x in `seq 2`; do
  [ "$1" == "--silence-label" ] && silence_label=$2 && shift 2;
  [ "$1" == "--cmd" ] && cmd="$2" && shift 2;
done

if [ $# != 3 ]; then
   echo "Word-align lattices (make the arcs sync up with words)"
   echo ""
   echo "Usage: $0 [options] <lang-dir> <decode-dir-in> <decode-dir-out>"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])] [--silence-label <integer-id-of-silence-word>]"
   exit 1;
fi

. ./path.sh || exit 1;

lang=$1
indir=$2
outdir=$3

mdl=`dirname $indir`/final.mdl
wbfile=$lang/phones/word_boundary.int

for f in $mdl $wbfile $indir/num_jobs; do
  [ ! -f $f ] && echo "word_align_lattices.sh: no such file $f" && exit 1;
done

mkdir -p $outdir/log


cp $indir/num_jobs $outdir;
nj=`cat $indir/num_jobs`

$cmd JOB=1:$nj $outdir/log/align.JOB.log \
  lattice-align-words --silence-label=$silence_label --test=true \
   $wbfile $mdl "ark:gunzip -c $indir/lat.JOB.gz|" "ark,t:|gzip -c >$outdir/lat.JOB.gz" || exit 1;

