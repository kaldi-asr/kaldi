#!/bin/bash

# Author: Hossein Hadian


[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <phones-dir> <ali-dir> <duration-model-dir>"
   echo "... where <duration-model-dir> is assumed to be a sub-directory "
   echo " of the directory where the transition model (i.e. final.mdl) is."
   echo "e.g.: $0 data/lang/phones exp/mono_ali exp/mono/durmod"
   echo ""
   exit 1;
fi

phonesdir=$1
alidir=$2
dir=$3
srcdir=`dirname $dir`

rm $dir/all.ali 2>&1 >/dev/null
mkdir -p $dir || exit 1;
gunzip -c $alidir/ali.*.gz >> $dir/all.ali || exit 1;

min_repeat_count=2
ali-to-phones --write-lengths $srcdir/final.mdl ark:$dir/all.ali ark,t:$dir/all.length_ali || exit 1;
max_duration=`python steps/durmod/find_max_duration.py $dir/all.length_ali $min_repeat_count | grep max-duration | awk '{print $2}'` || exit 1;
echo "Max duration: $max_duration" 
 
durmod-init --binary=true --max-duration=$max_duration --left-context=2 --right-context=1 \
             $phonesdir/roots.int $phonesdir/extra_questions.int $dir/0.dmdl || exit 1;
durmod-make-egs $dir/0.dmdl $srcdir/final.mdl ark:$dir/all.ali ark,t:$dir/egs.txt || exit 1;

echo "Done"
