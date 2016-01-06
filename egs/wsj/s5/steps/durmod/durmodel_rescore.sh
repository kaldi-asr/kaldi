#!/bin/bash

# Author: Hossein Hadian

lm_scale=1.0
cmd=run.pl

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <duration-model> <lattice-dir> <rescored-lattice-dir> (<phone-map>)"
   echo "... where <lattice-dir> is assumed to be a sub-directory "
   echo " of the directory where the transition model (i.e. final.mdl) is."
   echo "<phone-map> is for when the duration model has been trained using a different system (phones.txt is different)"
   echo "e.g.: $0 exp/mono/durmod exp/mono/decode_test_bg exp/mono/decode_test_bg_durmod"
   echo ""
   echo "  --lm-scale <float>                       # scale used for rescoring"   
   exit 1;
fi

durmodel=$1
latdir=$2
dir=$3
srcdir=`dirname $latdir`


nj=`cat $latdir/num_jobs`
mkdir -p $dir/log || exit 1;

durmodel_context_size=`durmod-info $durmodel 2>/dev/null | grep full-context-size | awk '{print $2}'` || exit 1;
echo "Duration Model Context Size: "$durmodel_context_size
  
$cmd JOB=1:$nj $dir/log/rescore.JOB.log \
lattice-align-phones --remove-epsilon=false \
                 $srcdir/final.mdl "ark:gunzip -c $latdir/lat.JOB.gz |" ark:- \| \
                 lattice-expand-ngram-phone --n=$durmodel_context_size \
                 $srcdir/final.mdl ark:- ark:- \| \
                 durmod-rescore-lattice --lm-scale=$lm_scale $durmodel $srcdir/final.mdl \
                 ark:- "ark,t:|gzip -c >$dir/lat.JOB.gz" || exit 1;

echo "Done"
