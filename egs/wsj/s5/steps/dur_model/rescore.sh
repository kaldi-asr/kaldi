#!/bin/bash

# Copyright 2015 Hossein Hadian
# Apache 2.0.
#
# This script rescores the lattices in a 'decode' directory and writes the 
# rescored lattices in a new directory. Rescoring is done using the
# log-likelihoods computed using the nnet-duration-model.


# Begin configuration section.
duration_model_scale=0.3             # The scale with which the duration model scores are added to lm-scores of a lattice
cmd=run.pl
avg_logprobs_file=

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <nnet3-duration-model> <lattice-dir> <rescored-lattice-dir>"
   echo "... where <lattice-dir> is assumed to be a sub-directory "
   echo " of the directory where the transition model (i.e. final.mdl) is."
   echo "e.g.: $0 exp/mono/durmod/30.mdl exp/mono/decode_test_bg exp/mono/decode_test_bg_durmod"
   echo ""
   echo "  --duration-model-scale <float>                       # scale used for rescoring"
   exit 1;
fi

nnet_durmodel=$1
latdir=$2
dir=$3
srcdir=`dirname $latdir`

for f in $nnet_durmodel $latdir/lat.1.gz $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: Required file not found: $f" && exit 1;
done

nj=`cat $latdir/num_jobs`
mkdir -p $dir/log || exit 1;

$cmd JOB=1:$nj $dir/log/rescore.JOB.log \
      lattice-align-phones --remove-epsilon=false \
      $srcdir/final.mdl "ark:gunzip -c $latdir/lat.JOB.gz |" ark:- \| \
      nnet3-durmodel-rescore-lattice --duration-model-scale=$duration_model_scale \
      --avg-logprobs-file=$avg_logprobs_file \
      $nnet_durmodel $srcdir/final.mdl \
      ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
